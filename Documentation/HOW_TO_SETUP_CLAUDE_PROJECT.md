# How to Set Up Your Claude Project

This guide explains how to create a Claude Project with the AI assistant instructions for your Personal AI Twin project.

## What is a Claude Project?

Claude Projects allow you to:
- Upload project files (documentation, code) for Claude to reference
- Set custom instructions that persist across conversations
- Have Claude maintain context about your project
- Get consistent, informed assistance throughout development

## Setup Steps

### 1. Create a New Claude Project

1. Go to [claude.ai](https://claude.ai)
2. Click on "Projects" in the sidebar
3. Click "Create Project"
4. Name it: **"Personal AI Twin Development"**
5. Add a description: "RAG-based personal knowledge management system with FastAPI, Ollama, and pgvector"

### 2. Upload Project Documentation

Upload these three key files to your Project:

**Required files**:
- ✅ `README.md` - Project overview and quick start
- ✅ `TECHNICAL_DESIGN.md` - Complete technical specification
- ✅ `SETUP_SUMMARY.md` - Setup guide and priorities

**Optional files** (add as you create them):
- `docker-compose.yml` - For Docker-related questions
- `.env.example` - For configuration questions
- `requirements.txt` - For dependency questions

**How to upload**:
- Click "Add content" in your Project
- Select "Upload files"
- Choose the markdown files
- Click "Upload"

### 3. Add Custom Instructions

Choose ONE of these instruction files based on your needs:

#### Option A: Full Instructions (Recommended)
**File**: `CLAUDE_PROJECT_INSTRUCTIONS.md`
- Most comprehensive
- Includes examples and patterns
- Best for detailed guidance
- ~8KB of instructions

**How to add**:
1. Open `CLAUDE_PROJECT_INSTRUCTIONS.md`
2. Copy the entire contents
3. In your Claude Project, click "Project Settings"
4. Paste into "Custom Instructions" field
5. Save

#### Option B: Condensed Instructions
**File**: `CLAUDE_PROJECT_INSTRUCTIONS_CONDENSED.md`
- Balanced version
- Key information only
- ~3KB of instructions

#### Option C: Ultra-Condensed (If character limit)
**File**: `CLAUDE_PROJECT_INSTRUCTIONS_COPYPASTE.txt`
- Minimal version
- Fits strict character limits
- ~1KB of instructions

### 4. Test Your Setup

Start a new conversation in your Project and try:

```
Hi! What should I work on first for the Personal AI Twin project?
```

Claude should:
- Reference the uploaded documentation
- Suggest starting with Phase 1 MVP tasks
- Provide specific guidance from TECHNICAL_DESIGN.md

### 5. Add Files as You Progress

As you build the project, upload new files to keep Claude informed:

**When you create**:
- `app/main.py` → Upload for FastAPI help
- `app/models/database.py` → Upload for DB questions
- `app/services/embeddings.py` → Upload for ML guidance
- `tests/unit/test_*.py` → Upload for testing help

**Keep it updated**:
- Re-upload files when you make significant changes
- Remove outdated versions
- Limit to ~20-30 key files (Claude has context limits)

## Best Practices

### DO ✅
- Keep documentation files up to date
- Upload code files you're actively working on
- Ask specific questions ("Review my embedding service code")
- Reference file names when asking questions
- Use project across multiple conversations

### DON'T ❌
- Upload hundreds of files (context limits)
- Include generated files (`__pycache__`, `.pyc`)
- Upload large binary files
- Forget to update files after major changes
- Mix unrelated projects in same Claude Project

## Example Workflows

### Workflow 1: Starting a New Component

**You**: "I need to implement the embedding service. What should it look like?"

**Claude will**:
- Reference TECHNICAL_DESIGN.md Section 4.3
- Provide full implementation with types
- Include test examples
- Suggest next steps

### Workflow 2: Code Review

**You**: "Review this code" [paste your code]

**Claude will**:
- Check against project architecture
- Verify type hints and docstrings
- Suggest improvements
- Point out missing error handling

### Workflow 3: Debugging

**You**: "This embedding service is throwing an error" [paste error]

**Claude will**:
- Diagnose the issue
- Reference relevant documentation
- Provide fixed code
- Explain the solution

### Workflow 4: Planning

**You**: "What should I work on next?"

**Claude will**:
- Check current phase in roadmap
- Review completed tasks
- Suggest next priority
- Explain why it's important

## Updating Your Project

### When to Update Instructions

Update custom instructions when:
- Project architecture changes significantly
- You move to a new phase (Phase 1 → Phase 2)
- New patterns or standards emerge
- You want to emphasize different aspects

### When to Update Documentation

Re-upload documentation when:
- You complete a major milestone
- Database schema changes
- API endpoints are added/modified
- Significant refactoring occurs

## Tips for Maximum Effectiveness

1. **Be Specific**: Instead of "help with database", say "help me implement the document_chunks table from TECHNICAL_DESIGN.md Section 3"

2. **Reference Docs**: Say "According to TECHNICAL_DESIGN.md, should I..." to ensure Claude checks the docs

3. **Iterate**: Start conversations with "continuing from our last session..." to maintain context

4. **Ask for Tests**: Always ask "Can you also write tests for this?" to maintain coverage

5. **Request Reviews**: Regularly ask "Does this align with our architecture?" to stay on track

6. **Track Progress**: Periodically ask "What's the status of Phase 1?" to review progress

## Troubleshooting

**Claude isn't referencing my docs**
- Verify files are uploaded to the Project (not just chat)
- Mention the file name explicitly: "Check TECHNICAL_DESIGN.md"
- Re-upload if file was updated

**Instructions aren't being followed**
- Check custom instructions are saved in Project Settings
- Try the condensed version if hitting character limits
- Remind Claude: "Remember, you're my project manager for this"

**Context seems lost between conversations**
- Make sure you're in the same Project
- Start with "Continuing the Personal AI Twin project..."
- Re-upload recently modified files

**Claude suggests wrong architecture**
- Say "That doesn't match TECHNICAL_DESIGN.md Section X"
- Ask Claude to check the documentation
- Verify the right files are uploaded

## Advanced: Multiple Conversations

You can run parallel conversations in the same Project:

- **Thread 1**: Frontend work (HTMX components)
- **Thread 2**: Backend work (API endpoints)  
- **Thread 3**: Database design
- **Thread 4**: Testing

Each conversation maintains context about the project while focusing on different aspects.

## Summary Checklist

Setup checklist:
- [ ] Create Claude Project
- [ ] Upload README.md, TECHNICAL_DESIGN.md, SETUP_SUMMARY.md
- [ ] Add custom instructions (choose version)
- [ ] Test with a question
- [ ] Start building!

As you develop:
- [ ] Upload new code files when working on them
- [ ] Re-upload updated documentation
- [ ] Use project across all related conversations
- [ ] Ask for reviews and suggestions
- [ ] Track progress through Claude

---

## Quick Start Commands

Copy-paste these to test your setup:

```
1. What's the tech stack for this project?

2. Show me the database schema for document_chunks

3. What should I work on first?

4. Generate the FastAPI main.py file

5. How do I implement the RAG pipeline?
```

If Claude answers all these accurately with references to your docs, your Project is set up correctly! 🎉

---

**Next**: Start building! Ask Claude: "Let's implement the first component from Phase 1. Where should we start?"
