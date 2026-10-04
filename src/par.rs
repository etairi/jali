//! Deterministic parallelism behind the `parallel` feature.
//!
//! Each helper returns what the sequential loop it stands for returns, whatever the number of
//! threads: the items are independent of each other, the results are collected in index order,
//! and where the loop stops at its first error, the error returned is the first in index order.
//! Without the feature the helpers are those loops. With it, work after the first error (or
//! after the first accepted attempt) may be done, but none of it is returned. Inputs shorter
//! than a helper's `min` run as the loop on the calling thread, where the work is too small to
//! share.
use crate::{Error, rand::ByteStream};
use std::sync::atomic::{AtomicU32, Ordering};
use zeroize::Zeroizing;

#[cfg(test)]
mod tests;

/// `(0..n).map(f).collect()`: every item, in index order; in parallel if `n >= min`.
pub(crate) fn map_min<T: Send>(
    n: usize,
    min: usize,
    f: impl Fn(usize) -> T + Sync + Send,
) -> Vec<T> {
    #[cfg(feature = "parallel")]
    if n >= min.max(2) {
        use rayon::prelude::*;
        return (0..n).into_par_iter().map(f).collect();
    }
    let _ = min;
    (0..n).map(f).collect()
}

/// [`map_min`] for every `n` of two or more.
pub(crate) fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    map_min(n, 2, f)
}

/// `(0..n).map(f).collect::<Result<Vec<_>, _>>()`: the items in index order, or the first
/// error in index order; in parallel if `n >= min`.
pub(crate) fn try_map_min<T: Send>(
    n: usize,
    min: usize,
    f: impl Fn(usize) -> Result<T, Error> + Sync + Send,
) -> Result<Vec<T>, Error> {
    #[cfg(feature = "parallel")]
    if n >= min.max(2) {
        return map_min(n, min, f).into_iter().collect();
    }
    let _ = min;
    (0..n).map(f).collect()
}

/// [`try_map_min`] for every `n` of two or more.
pub(crate) fn try_map<T: Send>(
    n: usize,
    f: impl Fn(usize) -> Result<T, Error> + Sync + Send,
) -> Result<Vec<T>, Error> {
    try_map_min(n, 2, f)
}

/// `for item in items { f(item)? }`; in parallel if there are at least `min` items, and then
/// every item is processed and the first error in order is returned.
pub(crate) fn try_for_each_min<T: Send>(
    items: Vec<T>,
    min: usize,
    f: impl Fn(T) -> Result<(), Error> + Sync + Send,
) -> Result<(), Error> {
    #[cfg(feature = "parallel")]
    if items.len() >= min.max(2) {
        use rayon::prelude::*;
        return items
            .into_par_iter()
            .map(f)
            .collect::<Vec<_>>()
            .into_iter()
            .collect();
    }
    let _ = min;
    items.into_iter().try_for_each(f)
}

/// The number of attempts of a rejection loop that run at once: the threads of the current
/// pool, at most 8, with the `parallel` feature, and 1 without.
pub(crate) fn speculation_width() -> u32 {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().clamp(1, 8) as u32
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Whether an attempt may stop early: an attempt of a smaller index in its batch has decided
/// (been accepted or failed), so the loop will not take this attempt's outcome.
pub(crate) struct Abandon<'a> {
    decided: &'a AtomicU32,
    attempt: u32,
}
impl Abandon<'_> {
    pub(crate) fn requested(&self) -> bool {
        self.decided.load(Ordering::Relaxed) < self.attempt
    }
}
#[cfg(test)]
impl Abandon<'static> {
    /// For an attempt run on its own, outside a batch: never requested.
    pub(crate) fn never() -> Self {
        static NONE_DECIDED: AtomicU32 = AtomicU32::new(u32::MAX);
        Abandon {
            decided: &NONE_DECIDED,
            attempt: 0,
        }
    }
}

/// The outcomes of a rejection loop's attempts $`0,1,\dots`$ below `limit`, computed `width`
/// at a time, in parallel with the `parallel` feature.
///
/// Attempt $`a`$ must be a function of $`a`$ and of its coins alone: the bytes
/// $`[Na,N(a+1))`$ of the coin stream, which the sequential loop reads in order, $`N`$ per
/// attempt. A batch reads the coins of all its attempts at once, the same bytes, and never
/// extends past `limit`. The caller takes the outcomes in order, and the first that is not a
/// rejection (`Ok(None)`), an acceptance or an error, decides, as in the sequential loop; at
/// most `width - 1` attempts after it are started, and discarded.
///
/// An attempt may return `Ok(None)` early when [`Abandon::requested`] says so. That happens
/// only after an attempt $`d<a`$ of the batch has decided, since the batch records only
/// deciding attempts; the caller then stops at $`d`$ or before, so it never takes the outcome
/// of an abandoned attempt.
pub(crate) struct Attempts<'s, T, S, F, const N: usize> {
    run: F,
    coins: &'s mut S,
    width: u32,
    limit: u32,
    next: u32,
    batch: std::vec::IntoIter<Result<Option<T>, Error>>,
}
impl<'s, T, S, F, const N: usize> Attempts<'s, T, S, F, N>
where
    T: Send,
    S: ByteStream,
    F: Fn(u32, &[u8; N], &Abandon<'_>) -> Result<Option<T>, Error> + Sync,
{
    pub(crate) fn new(width: u32, limit: u32, coins: &'s mut S, run: F) -> Self {
        Self {
            run,
            coins,
            width: width.max(1),
            limit,
            next: 0,
            batch: Vec::new().into_iter(),
        }
    }
    /// The outcome of `attempt`, which must be the attempt after the previous call's (the
    /// first is 0) and below the limit.
    pub(crate) fn outcome(&mut self, attempt: u32) -> Result<Option<T>, Error> {
        assert!(
            attempt == self.next && attempt < self.limit,
            "attempts are taken in order, below the limit"
        );
        if self.batch.as_slice().is_empty() {
            let count = self.width.min(self.limit - attempt) as usize;
            let mut coins = Zeroizing::new(vec![0u8; N * count]);
            self.coins.fill(&mut coins)?;
            // The smallest index of the batch whose outcome is not a rejection, so far.
            let decided = AtomicU32::new(u32::MAX);
            let (run, coins, decided) = (&self.run, &coins, &decided);
            self.batch = map(count, |j| {
                let attempt = attempt + j as u32;
                let bytes = coins[N * j..N * (j + 1)].try_into().expect("N bytes");
                let outcome = run(attempt, bytes, &Abandon { decided, attempt });
                if !matches!(outcome, Ok(None)) {
                    decided.fetch_min(attempt, Ordering::Relaxed);
                }
                outcome
            })
            .into_iter();
        }
        self.next += 1;
        self.batch
            .next()
            .expect("one outcome per attempt of the batch")
    }
}
