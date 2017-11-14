The Biotite package bundles popular tools in computational biology into an
unifying framework. It offers file I/O operations, analyses and manipulations
for biological sequence and structure data. Furthermore, the package provides
interfaces for popular biological databases and external software.

The internal structure and sequence representations are based on *NumPy*
`ndarrays`, taking the advantage of C-accelerated operations. Time consuming
operations that could not be vectorised are mostly implemented in *Cython* in
order to achieve C-accelerations in those places, too.

Additionally the package aims for simple usability and extensibility: The
objects representing structures and sequences can be indexed and scliced like
an `ndarray`. Even the actual internal `ndarrays` are easily accessible
allowing advanced users to implement their own algorithms upon the existing
types.