//! Test only: the absorb and squeeze calls of the transcripts on this thread while a recording
//! runs, for the tests of the IO patterns and of the proof layouts. The copies of the sponge
//! that `Transcript::binding` squeezes are not recorded.
use super::Protocol;
use std::cell::RefCell;

/// One call on a transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    /// `Transcript::new` with the protocol and the context.
    Start(Protocol, Vec<u8>),
    /// The bytes of a nonempty absorb.
    Absorb(Vec<u8>),
    /// The length of a squeeze.
    Squeeze(usize),
}

thread_local! {
    static LOG: RefCell<Option<Vec<Event>>> = const { RefCell::new(None) };
}

/// Run `f` and return its value with the calls it made on transcripts on this thread.
pub(crate) fn recording<T>(f: impl FnOnce() -> T) -> (T, Vec<Event>) {
    LOG.with(|log| *log.borrow_mut() = Some(Vec::new()));
    let out = f();
    let events = LOG.with(|log| log.borrow_mut().take()).unwrap_or_default();
    (out, events)
}

fn push(event: Event) {
    LOG.with(|log| {
        if let Some(events) = log.borrow_mut().as_mut() {
            events.push(event);
        }
    });
}

pub(crate) fn start(protocol: Protocol, context: &[u8]) {
    push(Event::Start(protocol, context.to_vec()));
}

pub(crate) fn absorb(bytes: &[u8]) {
    if !bytes.is_empty() {
        push(Event::Absorb(bytes.to_vec()));
    }
}

pub(crate) fn squeeze(len: usize) {
    push(Event::Squeeze(len));
}

/// The events with consecutive absorbs joined, which leave the sponge as one absorb of their
/// concatenation does: the IO pattern as a sequence of messages.
pub(crate) fn messages(events: &[Event]) -> Vec<Event> {
    let mut out: Vec<Event> = Vec::new();
    for event in events {
        match (out.last_mut(), event) {
            (Some(Event::Absorb(last)), Event::Absorb(bytes)) => last.extend_from_slice(bytes),
            (Some(Event::Squeeze(last)), Event::Squeeze(len)) => *last += len,
            _ => out.push(event.clone()),
        }
    }
    out
}
