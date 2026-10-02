//! The helpers against the sequential loops they stand for: speculative attempts with the
//! outcome of every attempt forced, and the first error of a map. With the `parallel` feature
//! every test also runs in pools of 1, 2, 3, 4 and 8 threads.
use super::*;
use crate::rand::{AesPrg, domain};
use std::sync::Mutex;

/// The forced outcome of one attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Forced {
    Reject,
    Accept,
    Fail,
}

/// The sequential loop's result: the index and value of the first attempt below `limit` that
/// is not a rejection, its error, or the restart limit.
fn sequential(plan: &[Forced], limit: u32) -> Result<(u32, u64), Error> {
    for attempt in 0..limit {
        match plan[attempt as usize] {
            Forced::Reject => {}
            Forced::Accept => return Ok((attempt, 1000 + u64::from(attempt))),
            Forced::Fail => return Err(Error::Index),
        }
    }
    Err(Error::RestartLimit)
}

/// What the loop of [`speculative`] did.
struct Run {
    result: Result<(u32, u64), Error>,
    /// The attempts started, in order.
    started: Vec<u32>,
    /// The attempts that stopped early, in order.
    abandoned: Vec<u32>,
}

/// The loop of `Abdlop::prove_core_speculative` over forced outcomes. Each attempt checks that
/// it received bytes `32a..32a+32` of the coin stream, works a little and looks for
/// abandonment before and after.
fn speculative(plan: &[Forced], limit: u32, width: u32) -> Run {
    let key = [7u8; 32];
    let mut expected = vec![0u8; 32 * plan.len()];
    AesPrg::new(&key, domain(3, 0)).fill(&mut expected).unwrap();
    let (started, abandoned) = (Mutex::new(Vec::new()), Mutex::new(Vec::new()));
    let mut coins = AesPrg::new(&key, domain(3, 0));
    let run = |a: u32, coins: &[u8; 32], abandon: &Abandon<'_>| {
        started.lock().unwrap().push(a);
        let a_ = a as usize;
        assert_eq!(
            coins[..],
            expected[32 * a_..32 * a_ + 32],
            "coins of attempt {a}"
        );
        for _ in 0..2 {
            if abandon.requested() {
                abandoned.lock().unwrap().push(a);
                return Ok(None);
            }
            // Some work, so that attempts overlap in a pool.
            let mut work = [0u8; 4096];
            AesPrg::new(&key, domain(9, a)).fill(&mut work).unwrap();
        }
        match plan[a_] {
            Forced::Reject => Ok(None),
            Forced::Accept => Ok(Some(1000 + u64::from(a))),
            Forced::Fail => Err(Error::Index),
        }
    };
    let mut attempts = Attempts::new(width, limit, &mut coins, run);
    let result = (|| {
        for attempt in 0..limit {
            if let Some(value) = attempts.outcome(attempt)? {
                return Ok((attempt, value));
            }
        }
        Err(Error::RestartLimit)
    })();
    let (mut started, mut abandoned) = (
        started.into_inner().unwrap(),
        abandoned.into_inner().unwrap(),
    );
    started.sort_unstable();
    abandoned.sort_unstable();
    Run {
        result,
        started,
        abandoned,
    }
}

/// Run `f` in pools of 1, 2, 3, 4 and 8 threads with the `parallel` feature, once without.
fn in_pools(f: impl Fn() + Send + Sync) {
    #[cfg(feature = "parallel")]
    for threads in [1, 2, 3, 4, 8] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(&f);
    }
    #[cfg(not(feature = "parallel"))]
    f();
}

/// Check a run against the sequential loop: the same result; whole batches of `width` started
/// up to the deciding attempt, never one at or beyond the limit; only attempts after the
/// deciding one abandoned, each after an attempt of its batch that decided.
fn check(plan: &[Forced], limit: u32, width: u32) {
    let run = speculative(plan, limit, width);
    let what = format!("{plan:?} limit {limit} width {width}");
    assert_eq!(run.result, sequential(plan, limit), "{what}");
    let deciding = plan[..limit as usize]
        .iter()
        .position(|f| *f != Forced::Reject)
        .map_or(limit.saturating_sub(1), |i| i as u32);
    let end = match limit {
        0 => 0,
        _ => (deciding / width + 1).saturating_mul(width).min(limit),
    };
    assert_eq!(run.started, (0..end).collect::<Vec<_>>(), "{what}");
    for a in &run.abandoned {
        let batch = a / width * width;
        assert!(
            *a > deciding && (batch..*a).any(|d| plan[d as usize] != Forced::Reject),
            "{what}: attempt {a} abandoned"
        );
    }
    // One thread runs a batch in order, so every attempt after the deciding one is abandoned.
    #[cfg(feature = "parallel")]
    let one_thread = rayon::current_num_threads() == 1;
    #[cfg(not(feature = "parallel"))]
    let one_thread = true;
    if one_thread && deciding < limit && plan[deciding as usize] != Forced::Reject {
        assert_eq!(
            run.abandoned,
            (deciding + 1..end).collect::<Vec<_>>(),
            "{what}"
        );
    }
}

#[test]
fn acceptance_forced_at_0_k_minus_1_k_and_the_limit_gives_the_sequential_result() {
    in_pools(|| {
        for width in [1, 2, 3, 4, 8, 16] {
            for limit in [1, 2, 5, 8, 9, 17, 33] {
                // Acceptance forced at attempt 0, K - 1, K and at the last attempt the limit
                // allows, every other attempt rejected; and acceptance only past the limit.
                let mut accepted = vec![0, width - 1, width, limit - 1, limit];
                accepted.sort_unstable();
                accepted.dedup();
                for j in accepted {
                    let mut plan = vec![Forced::Reject; limit.max(width) as usize + 1];
                    plan[j as usize] = Forced::Accept;
                    check(&plan, limit, width);
                }
            }
        }
    });
}

#[test]
fn the_first_acceptance_or_error_in_order_decides_within_and_across_batches() {
    use Forced::*;
    let plans: [&[Forced]; 8] = [
        &[Accept, Accept, Accept, Accept],
        &[Reject, Accept, Fail, Accept],
        &[Reject, Fail, Accept, Accept],
        &[Fail, Accept, Accept, Accept],
        &[Reject, Reject, Reject, Fail, Accept, Fail],
        &[Reject, Reject, Reject, Reject, Reject, Accept, Accept, Fail],
        &[Reject; 8],
        &[Reject, Reject, Reject, Reject, Fail, Reject, Reject, Reject],
    ];
    in_pools(|| {
        for plan in plans {
            for limit in 0..=plan.len() as u32 {
                for width in [1, 2, 3, 4, 8] {
                    check(plan, limit, width);
                }
            }
        }
    });
}

#[test]
#[should_panic(expected = "attempts are taken in order")]
fn attempts_out_of_order_are_refused() {
    let mut coins = AesPrg::new(&[7; 32], 0);
    let mut attempts = Attempts::new(2, 4, &mut coins, |_, _: &[u8; 32], _: &Abandon<'_>| {
        Ok(None::<u64>)
    });
    let _ = attempts.outcome(1);
}

#[test]
fn the_speculation_width_is_the_pool_size_capped_at_8() {
    #[cfg(feature = "parallel")]
    for (threads, width) in [(1, 1), (2, 2), (3, 3), (4, 4), (8, 8), (9, 8), (12, 8)] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(speculation_width), width);
    }
    #[cfg(not(feature = "parallel"))]
    assert_eq!(speculation_width(), 1);
}

#[test]
fn maps_keep_the_order_and_return_the_first_error_in_order() {
    in_pools(|| {
        for n in [0usize, 1, 2, 7, 64, 1000] {
            for min in [0, 2, 8, 64] {
                let squares: Vec<_> = (0..n).map(|i| i * i).collect();
                assert_eq!(map_min(n, min, |i| i * i), squares);
                assert_eq!(map(n, |i| i * i), squares);
                // Errors at every third index from `first` on, of two kinds.
                for first in [0, 1, n / 2, n.saturating_sub(1), n] {
                    let f = |i: usize| match i {
                        i if i >= first && (i - first) % 3 == 0 => Err(if i % 2 == 0 {
                            Error::Index
                        } else {
                            Error::Overflow
                        }),
                        i => Ok(i + 1),
                    };
                    let expected: Result<Vec<_>, _> = (0..n).map(f).collect();
                    assert_eq!(try_map_min(n, min, f), expected, "{n} {min} {first}");
                    assert_eq!(try_map(n, f), expected, "{n} {first}");
                    let seen = Mutex::new(Vec::new());
                    let each = try_for_each_min((0..n).collect(), min, |i| {
                        f(i)?;
                        seen.lock().unwrap().push(i);
                        Ok(())
                    });
                    assert_eq!(each, expected.as_ref().map(|_| ()).map_err(|e| e.clone()));
                    // Every item before the first error is processed.
                    let mut seen = seen.into_inner().unwrap();
                    seen.sort_unstable();
                    let before = (0..n).take_while(|i| f(*i).is_ok()).collect::<Vec<_>>();
                    assert_eq!(seen[..before.len()], before[..], "{n} {min} {first}");
                }
            }
        }
    });
}
