# agentsmd-guide.md

## How to Write High-Quality agents.md Files for GitHub Copilot Agents

This guide synthesizes best practices and lessons learned from thousands of real-world agents.md files, providing a robust framework for creating effective agent instructions. Use this as your ground truth for Copilot agent authoring.

---

### 1. Purpose of agents.md
- **agents.md** is a dedicated file for guiding AI coding agents, separate from README.md, focused on agent-specific context, commands, code style, boundaries, and project structure.
- It acts as a predictable source of truth for agents, ensuring consistent behavior and output.

### 2. Core Sections to Include
Successful agents.md files cover these six areas:
- **Agent Persona & Role:** Define the agent’s job, skills, and scope. Be specific (e.g., “QA engineer for unit tests in PyTest”).
- **Project Knowledge:** List tech stack (with versions), file structure, and key dependencies.
- **Executable Commands:** Provide exact commands (with flags/options) for build, test, lint, and other tasks. Prefer file-scoped commands for speed and safety.
- **Code Style & Examples:** Show real code snippets that demonstrate preferred style, naming conventions, and error handling. Examples are more effective than explanations.
- **Boundaries & Permissions:** Explicitly state what the agent can and cannot do. Use three tiers: Always do, Ask first, Never do.
- **Workflow & Standards:** Include PR checklists, test-first instructions, and any team-specific rules.

### 3. Best Practices
- **Be Specific:** Name technologies, versions, and file paths. Avoid vague instructions.
- **Concrete Examples:** Use real code and file references. Point to good and bad examples.
- **Boundaries:** Prevent destructive actions (e.g., “Never commit secrets”, “Do not edit src/”).
- **Commands Early:** List key commands near the top for quick agent reference.
- **Iterate:** Update agents.md as your project evolves. Add rules when mistakes repeat.
- **Keep It Concise:** Focus on actionable guidance. Avoid unnecessary verbosity.

### 4. Sample agents.md Template
```
---
name: your-agent-name
description: [One-sentence summary]
---

You are an expert [role] for this project.

## Persona
- You specialize in [task]
- You understand [project specifics]
- Your output: [expected deliverables]

## Project knowledge
- **Tech Stack:** [technologies, versions]
- **File Structure:**
  - `src/` – [source code]
  - `tests/` – [tests]
  - `docs/` – [documentation]

## Commands you can use
- Build: `npm run build`
- Test: `npm test`
- Lint: `npm run lint --fix`
- Typecheck: `npm run tsc --noEmit path/to/file.tsx`

## Code style
- Functions: camelCase
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE

### Example
```typescript
// Good
async function fetchUserById(id: string): Promise<User> {
  if (!id) throw new Error('User ID required');
  const response = await api.get(`/users/${id}`);
  return response.data;
}
// Bad
async function get(x) {
  return await api.get('/users/' + x).data;
}
```

## Boundaries
- ✅ **Always:** [actions allowed]
- ⚠️ **Ask first:** [actions needing approval]
- 🚫 **Never:** [forbidden actions]

## PR checklist
- Title format: `feat(scope): short description`
- Lint, type check, unit tests – all green before commit
- Diff is small and focused, with summary

## When stuck
- Ask a clarifying question, propose a plan, or open a draft PR with notes
```

### 5. Advanced Tips
- **File-Scoped Commands:** Prefer commands that operate on changed files (e.g., `npm run lint --fix path/to/file.tsx`).
- **Safety:** Specify which commands/actions require confirmation (e.g., installs, pushes, deletes).
- **Project Structure Index:** Point to key files (routes, tokens, main components) for faster agent context.
- **Design System:** Reference indexed docs and examples for UI consistency.
- **Test-First Mode:** Instruct agents to write/update tests before code changes for new features or bug fixes.
- **Nested agents.md:** For large repos, use agents.md in subdirectories for tailored guidance.

### 6. Common Pitfalls to Avoid
- Vague roles (“You are a helpful assistant”) – always specify scope and expertise.
- Missing boundaries – always state what the agent must not do.
- Lack of examples – always show preferred code and structure.
- Overly broad commands – prefer file-specific actions.

### 7. Iteration & Maintenance
- Treat agents.md as living documentation. Update as your project, stack, or team practices change.
- Use feedback from agent output to refine instructions and examples.

---

By following these guidelines, you ensure your agents.md files empower Copilot agents to deliver high-quality, context-aware, and safe automation tailored to your project’s needs.
