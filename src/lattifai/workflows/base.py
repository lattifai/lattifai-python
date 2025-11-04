"""
Base classes for agentic workflows
"""

import abc
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import colorful


def setup_workflow_logger(name: str) -> logging.Logger:
    """Setup a logger with consistent formatting for workflow modules"""
    logger = logging.getLogger(f"workflows.{name}")

    # Only add handler if it doesn't exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)+17s.py:%(lineno)-4d - %(levelname)-8s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


logger = setup_workflow_logger("base")


class WorkflowStatus(Enum):
    """Workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WorkflowResult:
    """Result of a workflow execution"""

    status: WorkflowStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    exception: Optional[Exception] = None  # Store the original exception object
    execution_time: Optional[float] = None
    step_results: Optional[List[Dict[str, Any]]] = None

    @property
    def is_success(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED

    @property
    def is_error(self) -> bool:
        return self.status == WorkflowStatus.FAILED


@dataclass
class WorkflowStep:
    """Individual step in a workflow"""

    name: str
    description: str
    required: bool = True
    retry_count: int = 0
    max_retries: int = 1

    def should_retry(self) -> bool:
        return self.retry_count < self.max_retries


class WorkflowAgent(abc.ABC):
    """Base class for agentic workflows"""

    def __init__(self, name: str, max_retries: int = 0):
        self.name = name
        self.max_retries = max_retries
        self.steps: List[WorkflowStep] = []
        self.logger = setup_workflow_logger("agent")

    @abc.abstractmethod
    def define_steps(self) -> List[WorkflowStep]:
        """Define the workflow steps"""
        pass

    @abc.abstractmethod
    async def execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a single workflow step"""
        pass

    def setup(self):
        """Setup the workflow"""
        self.steps = self.define_steps()
        for step in self.steps:
            step.max_retries = self.max_retries

    async def execute(self, **kwargs) -> WorkflowResult:
        """Execute the complete workflow"""
        if not self.steps:
            self.setup()

        start_time = time.time()
        context = kwargs.copy()
        step_results = []

        self.logger.info(colorful.bold_white_on_green(f"🚀 Starting workflow: {self.name}"))

        try:
            for i, step in enumerate(self.steps):
                step_info = f"📋 Step {i + 1}/{len(self.steps)}: {step.name}"
                self.logger.info(colorful.bold_white_on_green(step_info))

                step_start = time.time()
                step_result = await self._execute_step_with_retry(step, context)
                step_duration = time.time() - step_start

                step_results.append(
                    {"step_name": step.name, "status": "completed", "duration": step_duration, "result": step_result}
                )

                # Update context with step result
                context[f"step_{i}_result"] = step_result
                context[f'{step.name.lower().replace(" ", "_")}_result'] = step_result

                self.logger.info(f"✅ Step {i + 1} completed in {step_duration:.2f}s")

            execution_time = time.time() - start_time
            self.logger.info(f"🎉 Workflow completed in {execution_time:.2f}s")

            return WorkflowResult(
                status=WorkflowStatus.COMPLETED, data=context, execution_time=execution_time, step_results=step_results
            )

        except Exception as e:
            execution_time = time.time() - start_time
            # For LattifAI errors, just log the error code and basic message
            from lattifai.errors import LattifAIError

            if isinstance(e, LattifAIError):
                self.logger.error(f"❌ Workflow failed after {execution_time:.2f}s: [{e.error_code}] {e.message}")
            else:
                self.logger.error(f"❌ Workflow failed after {execution_time:.2f}s: {str(e)}")

            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=str(e),
                exception=e,  # Store the original exception
                execution_time=execution_time,
                step_results=step_results,
            )

    async def _execute_step_with_retry(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a step with retry logic"""
        last_error = None

        for attempt in range(step.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"🔄 Retrying step {step.name} (attempt {attempt + 1}/{step.max_retries + 1})")

                result = await self.execute_step(step, context)
                return result

            except Exception as e:
                last_error = e
                step.retry_count += 1

                # For LattifAI errors, show simplified message in logs
                from lattifai.errors import LattifAIError

                error_summary = f"[{e.error_code}]" if isinstance(e, LattifAIError) else str(e)[:100]

                if step.should_retry():
                    self.logger.warning(f"⚠️ Step {step.name} failed: {error_summary}. Retrying...")
                    continue
                else:
                    self.logger.error(
                        f"❌ Step {step.name} failed after {step.max_retries + 1} attempts: {error_summary}"
                    )
                    raise e

        # This should never be reached, but just in case
        if last_error:
            raise last_error
