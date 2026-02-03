#!/usr/bin/env python3
"""
Workflow State Manager

Manages workflow state for multi-agent orchestration.
Used by all agents to coordinate sequential execution.

Location: .github/agents_memory/workflow_state.json
"""

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, Optional, List

# Paths
AGENTS_MEMORY_DIR = Path(__file__).parent
STATE_FILE = AGENTS_MEMORY_DIR / "workflow_state.json"


class WorkflowState:
    """Workflow state manager for multi-agent coordination."""
    
    VALID_STAGES = [
        "1_datasets",
        "2_pipeline",
        "3_export",
        "4_push",
        "5_validate",
        "6_cleanup",
        "7_maintain",
        "8_finalize"
    ]
    
    VALID_STATUSES = ["pending", "running", "completed", "failed"]
    
    def __init__(self, workflow_id: Optional[str] = None):
        """Initialize workflow state.
        
        Args:
            workflow_id: Unique workflow ID (default: auto-generate)
        """
        self.workflow_id = workflow_id or self._generate_workflow_id()
        self.state = self._load_or_create()
    
    @staticmethod
    def _generate_workflow_id() -> str:
        """Generate unique workflow ID."""
        return f"wf_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    
    def _load_or_create(self) -> Dict[str, Any]:
        """Load existing state or create new."""
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        else:
            return self._create_default_state()
    
    def _create_default_state(self) -> Dict[str, Any]:
        """Create default workflow state."""
        return {
            "workflow_id": self.workflow_id,
            "status": "pending",
            "current_stage": 0,
            "stages": {stage: "pending" for stage in self.VALID_STAGES},
            "artifacts": {
                "datasets_count": 0,
                "datasets_dir": "data/datasets_v2",
                "pipeline_runs": 0,
                "training_examples": 0,
                "hf_space_url": None
            },
            "config": {
                "datasets": 500,
                "min_support": 3,
                "max_size": 3,
                "llm_model": "gpt-4.1-mini"
            },
            "started_at": None,
            "updated_at": None,
            "completed_at": None
        }
    
    def save(self):
        """Save state to file."""
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        AGENTS_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def start_workflow(self, config: Optional[Dict[str, Any]] = None):
        """Start a new workflow."""
        self.state["status"] = "running"
        self.state["started_at"] = datetime.now(UTC).isoformat()
        if config:
            self.state["config"].update(config)
        self.save()
    
    def complete_stage(self, stage: str, artifacts: Optional[Dict[str, Any]] = None):
        """Mark a stage as completed.
        
        Args:
            stage: Stage name (e.g., "1_datasets")
            artifacts: Artifact updates (e.g., {"datasets_count": 500})
        """
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.VALID_STAGES}")
        
        self.state["stages"][stage] = "completed"
        self.state["current_stage"] = int(stage.split("_")[0])
        
        if artifacts:
            self.state["artifacts"].update(artifacts)
        
        self.save()
    
    def fail_stage(self, stage: str, error: str):
        """Mark a stage as failed."""
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage: {stage}")
        
        self.state["stages"][stage] = "failed"
        self.state["status"] = "failed"
        self.state["error"] = error
        self.save()
    
    def complete_workflow(self):
        """Mark workflow as completed."""
        self.state["status"] = "completed"
        self.state["completed_at"] = datetime.now(UTC).isoformat()
        self.save()
    
    def get_next_stage(self) -> Optional[str]:
        """Get the next pending stage.
        
        Returns:
            Stage name or None if all stages complete
        """
        for stage in self.VALID_STAGES:
            if self.state["stages"][stage] == "pending":
                return stage
        return None
    
    def is_stage_complete(self, stage: str) -> bool:
        """Check if a stage is completed."""
        return self.state["stages"].get(stage) == "completed"
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        lines = []
        lines.append(f"Workflow ID: {self.state['workflow_id']}")
        lines.append(f"Status: {self.state['status']}")
        lines.append(f"Current Stage: {self.state['current_stage']}/8")
        lines.append("\nStages:")
        for stage, status in self.state["stages"].items():
            emoji = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}
            lines.append(f"  {emoji[status]} {stage}: {status}")
        lines.append(f"\nArtifacts:")
        for key, value in self.state["artifacts"].items():
            lines.append(f"  - {key}: {value}")
        return "\n".join(lines)


# Convenience functions

def load_workflow() -> WorkflowState:
    """Load current workflow state."""
    return WorkflowState()


def init_workflow(config: Optional[Dict[str, Any]] = None) -> WorkflowState:
    """Initialize new workflow.
    
    Args:
        config: Workflow config (datasets count, min_support, etc.)
    
    Returns:
        WorkflowState instance
    """
    wf = WorkflowState()
    wf.start_workflow(config)
    return wf


def complete_stage(stage: str, artifacts: Optional[Dict[str, Any]] = None):
    """Mark stage as completed (convenience function).
    
    Args:
        stage: Stage name (e.g., "1_datasets")
        artifacts: Artifact updates
    """
    wf = load_workflow()
    wf.complete_stage(stage, artifacts)


def get_workflow_status() -> str:
    """Get workflow status summary."""
    wf = load_workflow()
    return wf.get_status_summary()


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python workflow_state.py init [--datasets 500]")
        print("  python workflow_state.py complete <stage> [--count 500]")
        print("  python workflow_state.py status")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "init":
        config = {}
        if "--datasets" in sys.argv:
            idx = sys.argv.index("--datasets")
            config["datasets"] = int(sys.argv[idx + 1])
        wf = init_workflow(config)
        print(f"✅ Workflow initialized: {wf.workflow_id}")
        print(wf.get_status_summary())
    
    elif cmd == "complete":
        stage = sys.argv[2]
        artifacts = {}
        if "--count" in sys.argv:
            idx = sys.argv.index("--count")
            artifacts["datasets_count"] = int(sys.argv[idx + 1])
        complete_stage(stage, artifacts)
        print(f"✅ Stage {stage} completed")
        print(get_workflow_status())
    
    elif cmd == "status":
        print(get_workflow_status())
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
