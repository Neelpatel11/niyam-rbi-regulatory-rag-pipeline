# Fixture corpus

The documents in `corpus/` are synthetic. They copy the *structure* of regulatory directions
(chapters, numbered clauses, sub-clauses, tables, repeal provisions, a withdrawn direction and its
replacement) so that tests and the CI smoke evaluation exercise every code path without network
access.

They are not RBI text. They must not be cited, quoted or used as a source of regulatory
information. Their reference numbers start with `FIX` and their URLs use the reserved
`example.invalid` domain so they cannot be mistaken for real circulars.
