def highlight_text_with_arrows(input_text, start_pos, end_pos):
    if not start_pos or not end_pos:
        return "Error: Missing position information for highlighting."

    highlighted_result = ''
    try:
        line_start_idx = max(input_text.rfind('\n', 0, start_pos.idx), 0)
        line_end_idx = input_text.find('\n', line_start_idx + 1)
        if line_end_idx < 0:
            line_end_idx = len(input_text)

        total_lines = end_pos.ln - start_pos.ln + 1

        for line_number in range(total_lines):
            current_line = input_text[line_start_idx:line_end_idx]

            if line_number == 0:
                arrow_start_col = start_pos.col
            else:
                arrow_start_col = 0

            if line_number == total_lines - 1:
                arrow_end_col = end_pos.col
            else:
                arrow_end_col = len(current_line)

            highlighted_result += current_line + '\n'
            highlighted_result += ' ' * arrow_start_col + '^' * (arrow_end_col - arrow_start_col)

            line_start_idx = line_end_idx
            line_end_idx = input_text.find('\n', line_start_idx + 1)
            if line_end_idx < 0:
                line_end_idx = len(input_text)

        return highlighted_result.replace('\t', '')
    except AttributeError:
        return "Error: Invalid start or end position attributes."
