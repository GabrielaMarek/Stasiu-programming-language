def highlight_text_with_arrows(input_text, start_pos, end_pos):
    highlighted_result = ''

    line_start_idx = max(input_text.rfind('\n', 0, start_pos.idx), 0)
    line_end_idx = input_text.find('\n', line_start_idx + 1)
    if line_end_idx < 0: 
        line_end_idx = len(input_text)
    
    total_lines = end_pos.ln - start_pos.ln + 1 

    for line_number in range(total_lines):
        current_line = input_text[line_start_idx:line_end_idx]

        arrow_start_col = start_pos.col if line_number == 0 else 0  
        arrow_end_col = end_pos.col if line_number == total_lines - 1 else len(current_line) - 1 

        highlighted_result += current_line + '\n'
        highlighted_result += ' ' * arrow_start_col + '^' * (arrow_end_col - arrow_start_col)

        line_start_idx = line_end_idx
        line_end_idx = input_text.find('\n', line_start_idx + 1)
        if line_end_idx < 0: 
            line_end_idx = len(input_text)

    return highlighted_result.replace('\t', '')
