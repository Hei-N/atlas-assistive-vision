"""Camera-motion estimation (Atlas Phase 4).

Contains a generic MotionEstimator interface and concrete estimator
implementations. Downstream code should depend on the MotionEstimator
abstraction and the MotionEstimate result type (src/models.py), not on any
specific estimator implementation.
"""
