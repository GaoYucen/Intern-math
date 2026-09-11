"""Competition entry with explicit, environment-independent R2 defaults."""
from r2_agent import ReasoningAgent as _R2, R2Config

class SubmissionConfig(R2Config):
    @property
    def mode(self):
        return 'r2_budgeted_tools'

class ReasoningAgent(_R2):
    def __init__(self, client, *args, **kwargs):
        if kwargs.get('config') is None:
            kwargs['config'] = SubmissionConfig()
        super().__init__(client, *args, **kwargs)

__all__ = ['ReasoningAgent']
