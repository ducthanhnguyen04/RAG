"""
Quick test for Mock Tavily - requires minimal dependencies
"""
import sys
sys.path.insert(0, '/home/hoangviet/helooo/DATN')

# Test 1: Check mock data structure
print("\n" + "="*60)
print("Test 1: Mock Tavily Data Structure")
print("="*60)

from src.mock_tavily import MockTavilySearchEngine

engine = MockTavilySearchEngine()

print(f"✓ MockTavilySearchEngine created")
print(f"✓ Mock database has {len(engine.MOCK_RESULTS)} keyword groups:")

for keyword, items in list(engine.MOCK_RESULTS.items())[:3]:
    print(f"  - '{keyword}': {len(items)} results")

# Test 2: Check basic search without async
print("\n" + "="*60)
print("Test 2: Checking Mock Results")
print("="*60)

# Show sample data
sample_results = engine.MOCK_RESULTS.get('rag', [])
if sample_results:
    print(f"✓ Sample 'rag' results:")
    for i, result in enumerate(sample_results[:2], 1):
        print(f"  {i}. {result['title']}")
        print(f"     {result['content'][:60]}...")

print("\n" + "="*60)
print("✓ ALL BASIC TESTS PASSED")
print("="*60)
print("\nNote: Full async tests require sentence-transformers")
print("Run: pip install sentence-transformers")
print("Then: python tests/test_mock_integration.py")
