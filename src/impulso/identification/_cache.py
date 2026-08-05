"""Posterior-identity memo shared by identification schemes."""

import weakref
from typing import Any, Final

_CACHE_MISS: Final = object()
"""Sentinel returned by `_PosteriorCache.get` when there is no valid entry.

A dedicated sentinel (rather than `None`) keeps `None` usable as a cached
value, and makes the hit test at call sites an unambiguous `is` check.
"""


class _PosteriorCache:
    """Single-slot memo keyed on object identity, validated by weak references.

    Several identification schemes memoise a quantity that depends only on
    the posterior (plus a few scalars) so that a per-period identification
    loop — `IdentifiedVAR._identify_per_t` calls `identify()` once per time
    index with the same posterior — pays the expensive part once instead of
    `T` times.

    Keying such a memo on `id(posterior)` is unsafe: once the posterior is
    garbage collected its address can be reused by a new object, and a
    subsequent lookup with a *different* posterior that happens to land on
    the recycled address returns a stale value silently (issue #203). This
    cache stores `weakref.ref(owner)` instead and treats a dead referent —
    or a live referent that is not the object being looked up — as a miss.
    The referent identity check (`ref() is owner`) subsumes an `id()`
    comparison exactly, so no `id()` is kept in the key.

    Usage is a get/compute/set triple:

        cached = self._cache.get(posterior, (n_lags,))
        if cached is _CACHE_MISS:
            cached = expensive(posterior, n_lags)
            self._cache.set(posterior, (n_lags,), cached)

    `owners` may be a single object or a tuple of objects when validity
    depends on more than one identity (`ProxySVAR` keys on the posterior
    *and* the `VARData`). Every owner must support weak references —
    `xr.Dataset` and Pydantic models such as `VARData` both do. If any
    owner does not (plain tuples, ints and strings do not), `set` declines
    to cache rather than falling back to an unsafe key, so the only cost is
    a lost speed-up.

    Entries in `key` are ordinary scalars compared with `==` (lag order, a
    variable name, a horizon). Do not put arrays there — elementwise
    comparison would not yield a bool.

    Adopted by `ProxySVAR._impact_cache` and `LongRunRestriction._lr_cache`
    (issue #203). The one remaining identity-keyed memo in this module
    collapses to the same three lines once its branch merges:

        MaxShare._spectral_cache:
            self._spectral_cache.get(posterior, (n_lags, target))
            self._spectral_cache.set(posterior, (n_lags, target), value)

    Declare the attribute on the (frozen) scheme with
    `PrivateAttr(default_factory=_PosteriorCache)` so each instance gets
    its own slot, and mutate it in place — no `object.__setattr__` needed.
    """

    __slots__ = ("_key", "_refs", "_value")

    def __init__(self) -> None:
        self._refs: tuple[weakref.ref, ...] | None = None
        self._key: tuple[Any, ...] = ()
        self._value: Any = _CACHE_MISS

    @staticmethod
    def _as_tuple(owners: Any) -> tuple[Any, ...]:
        """Normalise a single owner or a tuple of owners to a tuple."""
        return owners if isinstance(owners, tuple) else (owners,)

    def get(self, owners: Any, key: tuple[Any, ...] = ()) -> Any:
        """Look up the memoised value.

        Args:
            owners: The object (or tuple of objects) whose identity the
                cached value is tied to.
            key: Scalar key tail — everything the value depends on that is
                not an owner identity.

        Returns:
            The cached value, or `_CACHE_MISS` if there is no entry, the
            key tail differs, the owners differ, or any owner has been
            garbage collected.
        """
        if self._refs is None or self._key != key:
            return _CACHE_MISS
        owner_tuple = self._as_tuple(owners)
        if len(owner_tuple) != len(self._refs):
            return _CACHE_MISS
        # A dead referent dereferences to None, which can never be an
        # owner (weakref.ref(None) is not constructible), so the same
        # check covers both "collected" and "different object".
        if any(ref() is not owner for ref, owner in zip(self._refs, owner_tuple, strict=True)):
            return _CACHE_MISS
        return self._value

    def set(self, owners: Any, key: tuple[Any, ...], value: Any) -> None:
        """Store `value`, tying its validity to the owners staying alive.

        Args:
            owners: The object (or tuple of objects) whose identity the
                value depends on.
            key: Scalar key tail.
            value: The value to memoise.
        """
        try:
            refs = tuple(weakref.ref(owner) for owner in self._as_tuple(owners))
        except TypeError:
            # An owner that cannot be weakly referenced has no validity
            # token, so caching it would reintroduce the id()-reuse bug.
            self.clear()
            return
        self._refs = refs
        self._key = key
        self._value = value

    def clear(self) -> None:
        """Drop any stored entry."""
        self._refs = None
        self._key = ()
        self._value = _CACHE_MISS
