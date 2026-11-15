"""Hosted GPU runs: ``ii run --remote``.

``client`` drives a run from this machine, ``worker`` is what runs next to the GPU, ``backends``
says where that is (Modal, or in-process for tests), and ``pricing`` keeps the cost in view.
Only ``modal_app`` imports Modal, and only when a Modal run starts.
"""
