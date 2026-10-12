"""Neural architecture components."""

from .model import DockingWizardModel, DockingWizardModelConfig

LangProModel = DockingWizardModel
LangProModelConfig = DockingWizardModelConfig

__all__ = [
    "LangProModel",
    "LangProModelConfig",
    "DockingWizardModel",
    "DockingWizardModelConfig",
]

__all__ = ["DockingWizardModel", "DockingWizardModelConfig"]
