import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ProjectPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        cls.skill = (
            ROOT / "docs" / "skills" / "c3da-experiment-workflow" / "SKILL.md"
        ).read_text(encoding="utf-8")

    def test_agents_requires_project_skill(self):
        self.assertIn(
            "docs/skills/c3da-experiment-workflow/SKILL.md", self.agents
        )

    def test_skill_requires_new_branch_before_changes(self):
        self.assertIn("修改前创建新分支", self.skill)

    def test_skill_forbids_cross_run_artifact_reuse(self):
        self.assertIn("禁止跨运行复用或混合产物", self.skill)

    def test_skill_requires_full_command_and_hash_records(self):
        self.assertIn("完整训练命令", self.skill)
        self.assertIn("SHA256", self.skill)

    def test_skill_keeps_master_as_verified_best_only(self):
        self.assertIn("master", self.skill)
        self.assertIn("当前最佳", self.skill)

    def test_skill_distinguishes_observations_from_selection_limits(self):
        self.assertIn("黄金观察值", self.skill)
        self.assertIn("筛选配额", self.skill)


if __name__ == "__main__":
    unittest.main()
