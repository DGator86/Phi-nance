# Paperclip / OpenClaw CEO setup

This guide matches the flow in your screenshots: create a **CEO** agent in the OpenClaw-style UI, point it at this repo, and give it the Paperclip CEO persona files.

## What you are trying to build

You are creating a top-level **CEO** agent that:

- runs locally with **Codex** or **Claude Code**,
- uses this repo as its working directory,
- loads the Paperclip CEO instructions from `agents/ceo/AGENTS.md`, and
- starts with a task that tells it to create its heartbeat, then hire a founding engineer.

## Fastest setup path

From the repo root, run:

```bash
python scripts/bootstrap_paperclip_ceo.py
```

That command will:

- create `agents/ceo/`,
- download `AGENTS.md`, `HEARTBEAT.md`, `SOUL.md`, and `TOOLS.md` from the Paperclip CEO template, and
- print the exact paths you should paste into the UI.

If you need to replace files later:

```bash
python scripts/bootstrap_paperclip_ceo.py --force
```

## Files the script creates

After bootstrapping, you should have:

```text
agents/
  ceo/
    AGENTS.md
    HEARTBEAT.md
    SOUL.md
    TOOLS.md
```

These files come from the Paperclip CEO template repo.

## What to enter in the UI

### Agent screen

Use these values for the form shown in your screenshots.

- **Agent name:** `CEO`
- **Reports to:** leave empty if this is your top-level company agent
- **Adapter type:** `Codex (local)` is the cleanest choice if Codex is installed on the box
  - Use `Claude Code (local)` if that is the CLI you already have working
  - Do **not** use `Process` unless you want to manually wire a raw command and args
- **Working directory:** the absolute path to this repo
  - Example: `/workspace/Phi-nance`
- **Agent instructions file:** the absolute path to the downloaded CEO `AGENTS.md`
  - Example: `/workspace/Phi-nance/agents/ceo/AGENTS.md`
- **Prompt template:** keep the default template unless you have a reason to customize heartbeat framing

### Adapter environment check

Before clicking **Next**, use **Test environment** / **Test now**.

A passing check means the adapter CLI is installed and callable from the server where OpenClaw is running.

If the check fails:

- confirm `codex` or `claude` is installed on the machine,
- confirm the OpenClaw service can see that binary on `PATH`, and
- confirm the repo path exists on that same machine.

## What to put in the first task

Use the same intent shown in the screenshots, but slightly tightened up.

### Task title

```text
Create your CEO HEARTBEAT.md
```

### Description

```text
Set yourself up as the CEO.

Use the CEO persona files from the Paperclip template in agents/ceo:
https://github.com/paperclipai/companies/blob/main/default/ceo/AGENTS.md

Confirm that AGENTS.md, HEARTBEAT.md, SOUL.md, and TOOLS.md exist in agents/ceo and that AGENTS.md is the configured instructions file.

Then hire a Founding Engineer agent and start planning the first roadmap and task list for this company.
```

## Why `Process` is usually the wrong choice here

Your first screenshot shows **Process** selected. That option is only for manually specifying a command like:

- command: `python`
- args: `some_script.py,--flag`

For this use case, **Codex (local)** or **Claude Code (local)** is better because the adapter already knows how to run an interactive coding agent in the repo.

## Recommended adapter choice

### Pick `Codex (local)` if:

- you already have Codex installed,
- you want repo editing, shell execution, and iterative coding flow,
- you want the simplest mapping to the screenshots you shared.

### Pick `Claude Code (local)` if:

- that CLI is the one you already use daily,
- it passes the adapter environment check without extra setup.

## Minimal install checklist on the host machine

On the same machine that runs OpenClaw, verify:

```bash
pwd
python --version
which codex
```

If you plan to use Claude Code instead, verify:

```bash
which claude
```

Also verify the repo and CEO files exist:

```bash
test -d /workspace/Phi-nance && echo repo-ok
test -f /workspace/Phi-nance/agents/ceo/AGENTS.md && echo agents-ok
```

## Common mistakes

### 1. Using `Process` instead of `Codex`

If you choose `Process`, the form expects a raw executable and argument list. That is not what the Paperclip post is trying to show.

### 2. Using a relative path in the instructions file field

Use the full absolute path, not `agents/ceo/AGENTS.md`.

### 3. Forgetting sibling files

The CEO `AGENTS.md` explicitly references these files:

- `HEARTBEAT.md`
- `SOUL.md`
- `TOOLS.md`

If you only download `AGENTS.md`, the agent will start incomplete.

### 4. Setting the wrong working directory

The working directory should be the repo root, not `agents/ceo`.

## Suggested next agent after CEO

Once the CEO starts cleanly, create a second agent like:

- **Agent name:** `Founding Engineer`
- **Reports to:** `CEO`
- **Adapter:** `Codex (local)`
- **Working directory:** `/workspace/Phi-nance`

Then give it tasks such as:

- audit the current app state,
- identify the fastest path to a usable MVP,
- break roadmap items into implementation tasks.

## One-command bootstrap plus verification

If you want the full quick check:

```bash
python scripts/bootstrap_paperclip_ceo.py && test -f agents/ceo/AGENTS.md && echo ready
```

If that prints `ready`, you have the local file side set up correctly and can finish the rest in the UI.
