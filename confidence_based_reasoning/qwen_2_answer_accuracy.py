import re
import sys

def analyze(filepath="out.txt"):
    with open(filepath) as f:
        text = f.read()

    blocks = re.split(r"-{20,}", text)

    total = 0
    correct = 0
    reasoning_invoked = 0
    reasoning_correct = 0
    no_reasoning = 0
    no_reasoning_correct = 0
    reasoning_changed_answer = 0

    for i, block in enumerate(blocks):
        if i > 426: break
        block = block.strip()
        if not block:
            continue

        result_match = re.search(r"FINAL RESULT: \w+ \| (CORRECT|WRONG)", block)
        if not result_match:
            continue

        total += 1
        is_correct = result_match.group(1) == "CORRECT"
        if is_correct:
            correct += 1

        top_choice = re.search(r"Top choice: (\w+)", block)
        reasoning_result = re.search(r"Reasoning Result: (\w+)", block)
        invoked = "Invoking Reasoning Mode" in block

        if invoked:
            reasoning_invoked += 1
            if is_correct:
                reasoning_correct += 1
            if top_choice and reasoning_result and top_choice.group(1) != reasoning_result.group(1):
                reasoning_changed_answer += 1
        else:
            no_reasoning += 1
            if is_correct:
                no_reasoning_correct += 1

    print(f"Total questions:              {total}")
    print(f"Total correct:                {correct}")
    print(f"Overall accuracy:             {correct}/{total} = {correct/total*100:.2f}%")
    print()
    print(f"Reasoning invoked:            {reasoning_invoked}/{total} ({reasoning_invoked/total*100:.2f}%)")
    print(f"Reasoning accuracy:           {reasoning_correct}/{reasoning_invoked} = {reasoning_correct/reasoning_invoked*100:.2f}%")
    print()
    print(f"No reasoning (high conf):     {no_reasoning}/{total} ({no_reasoning/total*100:.2f}%)")
    print(f"No-reasoning accuracy:        {no_reasoning_correct}/{no_reasoning} = {no_reasoning_correct/no_reasoning*100:.2f}%")
    print()
    print(f"Reasoning changed answer:     {reasoning_changed_answer}/{reasoning_invoked}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "experiment_logs.txt"
    analyze(path)
