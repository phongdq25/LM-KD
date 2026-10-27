"""Reference backends used by the baseline reproduction (``lmkd.reproduce``): ``kto_runner`` and ``po_reference_runner``.

They run an audited third-party trainer end to end; they are not part of the LM-KD training loop. To use LM-KD loss
math inside your own HF/TRL trainer, subclass the trainer and call :mod:`lmkd.losses.functional` from its
``compute_loss``.
"""
