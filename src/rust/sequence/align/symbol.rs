//! The [`Symbol`] trait abstracts over the unsigned integer types used to
//! represent sequence codes, so the alignment algorithms can support different
//! alphabet sizes (`uint8`/`uint16`/`uint32`/`uint64` on the Python side).

/// A symbol code.
///
/// Implemented for the unsigned integer types.
pub trait Symbol: Copy + Eq {
    /// Convert the symbol code into an index usable for matrix lookup.
    fn index(self) -> usize;
}

macro_rules! impl_symbol {
    ($($t:ty),+ $(,)?) => {
        $(
            impl Symbol for $t {
                #[inline(always)]
                fn index(self) -> usize {
                    self as usize
                }
            }
        )+
    };
}

impl_symbol!(u8, u16, u32, u64);
