import re
import pandas as pd

def analyze_tuned_logs(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Split the file by the dashed separators
    blocks = content.split('----------------------------------------')
    
    data = []
    for block in blocks:
        if "Analysis:" not in block:
            continue
            
        # Extract ID, Initial Choice, Log-Diff, and Token Count
        q_id = re.search(r"Q(\d+)", block)
        initial = re.search(r"Top choice: ([A-D])", block)
        log_diff = re.search(r"Initial Log-Diff: ([\d\.]+)", block)
        tokens = re.search(r"Total output tokens: (\d+)", block)
        
        # Extract Final Result and Ground Truth
        # Handles both "FINAL RESULT: A | CORRECT" and "FINAL RESULT: C | WRONG (Target: D)"
        final_info = re.search(r"FINAL RESULT: ([A-D]) \| (CORRECT|WRONG)", block)
        
        # Check if Reasoning was invoked and what it decided
        used_reasoning = "Invoking Reasoning Mode" in block
        reasoning_choice = re.search(r"Reasoning Result: ([A-D])", block)

        if q_id and initial and final_info:
            # Determine Ground Truth
            final_pred = final_info.group(1)
            is_correct = final_info.group(2) == "CORRECT"
            
            if is_correct:
                target = final_pred
            else:
                target_match = re.search(r"Target: ([A-D])", block)
                target = target_match.group(1) if target_match else None

            # Logic for improved/hurt/changed
            improved = False
            hurt = False
            changed = False
            
            if used_reasoning and reasoning_choice:
                r_choice = reasoning_choice.group(1)
                changed = (initial.group(1) != r_choice)
                
                # Win: Initial was wrong, but final (after reasoning) is correct
                if initial.group(1) != target and is_correct:
                    improved = True
                # Loss: Initial was right, but final (after reasoning) is wrong
                elif initial.group(1) == target and not is_correct:
                    hurt = True

            data.append({
                "id": q_id.group(1),
                "used_reasoning": used_reasoning,
                "improved": improved,
                "hurt": hurt,
                "tokens": int(tokens.group(1)) if tokens else 0,
                "is_correct": is_correct,
                "changed": changed
            })

    df = pd.DataFrame(data)
    
    # --- CALCULATIONS ---
    total_q = len(df)
    reasoning_df = df[df['used_reasoning']]
    num_reasoning = len(reasoning_df)
    
    win_count = df['improved'].sum()
    fail_count = df['hurt'].sum()
    change_count = df['changed'].sum()
    
    improvement_rate = (win_count / num_reasoning) * 100 if num_reasoning > 0 else 0
    damage_rate = (fail_count / num_reasoning) * 100 if num_reasoning > 0 else 0
    avg_tokens = df['tokens'].mean()

    print("--- FIXED TUNED LOG ANALYSIS ---")
    print(f"Total Questions analyzed: {total_q}")
    print(f"Reasoning Trigger Rate:   {(num_reasoning/total_q)*100:.2f}% ({num_reasoning} questions)")
    print(f"Average Tokens/Question:  {avg_tokens:.2f} tokens")
    print("-" * 35)
    print(f"Reasoning Changed Answer: {change_count} times")
    print(f"Reasoning 'Win' Rate:     {improvement_rate:.2f}% ({win_count} corrected)")
    print(f"Reasoning 'Fail' Rate:    {damage_rate:.2f}% ({fail_count} ruined)")
    print(f"Net Accuracy Gain:        {(improvement_rate - damage_rate):.2f}%")
    
    return df

df = analyze_tuned_logs("out-tuned.txt")