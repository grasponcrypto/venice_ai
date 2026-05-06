# Task Plan - Address Major Issues from Code Review

## Current Status of Critical Issues
- [x] CRIT-1: Coordinator first refresh (already fixed in code)
- [x] CRIT-2: client.close() properly awaited in async_unload_entry (already fixed)
- [x] CRIT-3: Service handler calls public async_generate_data (already fixed)
- [x] CRIT-4: async_migrate_entry is @staticmethod with correct signature (already fixed)

## Major Issues to Fix (commit after each)
- [ ] MAJ-5: GenDataTask constructor uses wrong kwarg `task=` → should be `instructions=`
- [ ] MAJ-4: Broad `except Exception` swallows errors silently in repairs setup/unload
- [ ] MAJ-3: Hard-coded `conversation_id="service_call"` → use uuid4
- [ ] MAJ-1: Options flow creates new HTTP session on every open
- [ ] MAJ-2: VeniceAIOptionsFlow manually stores `config_entry` (shadows base class)
- [ ] MAJ-6: TTS entity ignores config entry options at call-time
- [ ] MAJ-7: Unused runtime_data import in tts.py
- [ ] GAP-1: Missing translations/en.json file
- [ ] GAP-3: Verify coordinator repairs listener cleanup on unload
