# Limitations

**regulo** is a transparent, reproducible reference implementation.
It is intentionally narrow in scope.  The following are explicit
non-goals:

1. **Layer-0 only for geometry-aware penalties.**  ``Covridge``
   and ``Sparridge`` apply only to the first weight matrix because
   the empirical Gram matrix ``C_{delta,n}`` is defined over the
   input dimension.  Users who want geometry-aware shrinkage on
   deeper layers must construct their own Gram matrix.

2. **No built-in real-data loaders.**  Earlier versions of this
   package shipped loaders for UCI Energy Efficiency and GSE9476
   Leukemia.  These have been removed: the library is now pure
   NumPy + SciPy with no pandas / scikit-learn dependency.  Users
   should bring their own arrays and pass them through
   :class:`regulo.tune.Scaler` for standardization.

3. **No GPU acceleration.**  All computation runs on CPU via
   NumPy.  For large datasets or architectures the per-epoch cost
   grows accordingly.  Use ``CuPy`` or ``PyTorch`` for production
   throughput.

4. **Pure NumPy back-propagation.**  Every gradient flows through
   hand-written code; there is no autograd engine.  The benefit
   is complete transparency and inspectability; the cost is that
   new layer types must be implemented by hand.

5. **Single-machine training.**  No distributed training, no
   model parallelism, no asynchronous optimisation.

6. **No model versioning beyond ``__version__``.**  ``save`` /
   ``load`` rejects mismatched **major** versions; minor-version
   drift is permitted.

7. **No hyperparameter search beyond grid + K-fold.**  No
   Bayesian optimisation, no population-based training, no
   learning-rate schedules.
