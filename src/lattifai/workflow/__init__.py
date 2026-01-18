"""LattifAI Agentic Workflows.

This module provides agentic workflow capabilities for automated processing
of multimedia content through intelligent agent-based pipelines.

Key Components:
    WorkflowAgent: Abstract base class for implementing workflow agents.
        Provides step-based execution with retry logic, state management,
        and consistent logging.

    WorkflowStep: Defines individual workflow steps with timing and
        execution status tracking.

    WorkflowResult: Encapsulates workflow execution results including
        status, outputs, errors, and timing information.

    FileExistenceManager: Handles file existence conflicts during workflows,
        supporting interactive and automatic resolution modes.

Example:
    >>> from lattifai.workflow import WorkflowAgent, WorkflowStep, WorkflowResult
    >>> class MyWorkflow(WorkflowAgent):
    ...     def define_steps(self):
    ...         return [WorkflowStep("download"), WorkflowStep("process")]
    ...     def execute_step(self, step, context):
    ...         # Implementation
    ...         pass

See Also:
    - lattifai.client.LattifAI: Main client that orchestrates workflows
    - lattifai.youtube: YouTube-specific workflow integration
"""

# Import transcript processing functionality


from .base import WorkflowAgent, WorkflowResult, WorkflowStep
from .file_manager import TRANSCRIBE_CHOICE, FileExistenceManager

__all__ = [
    "WorkflowAgent",
    "WorkflowStep",
    "WorkflowResult",
    "FileExistenceManager",
    "TRANSCRIBE_CHOICE",
]
