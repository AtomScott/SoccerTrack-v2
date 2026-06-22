"""Dataset utilities for SoccerTrack v2.

Made an explicit package (rather than a PEP 420 namespace package) so that
``src.data_utils.soccertrack_v2`` — on the BAS evaluation import path
(``src.evaluation.bas_map``) — resolves reliably regardless of how ``src`` is
placed on ``sys.path``.
"""
