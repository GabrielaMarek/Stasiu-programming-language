from highlight_text_with_arrows import *

#ERRORS

class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def stringify(self):
        result  = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.filename}, line {self.pos_start.ln + 1}'
        result += '\n\n' + highlight_text_with_arrows(self.pos_start.fulltxt, self.pos_start, self.pos_end)
        return result

class WrongSyntaxError(Error):
	def __init__(self, pos_start, pos_end, details=''):
			super().__init__(pos_start, pos_end, 'Oh no... Invalid Syntax...', details)

class CharacterFormatError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Oops! It looks like there is a character that is ILLEGAAAAAAAL!', details)

class RuntimeError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)
        self.context = context

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}\n'
        result += '\n' + highlight_text_with_arrows(self.pos_start.fulltxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  File {pos.filename}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result

#POSITION

class Position:
    def __init__(self, idx, ln, col, filename, fulltxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.filename = filename
        self.fulltxt = fulltxt

    def next_character(self, character_current):
        self.idx += 1
        self.col += 1

        if character_current == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.filename, self.fulltxt)

#CONSTANTS

DIGITS = '0123456789'
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'

#TOKENS

# Arithmetic Operators
TOKTYPE_PLUS       = 'PLUS'
TOKTYPE_MINUS      = 'MINUS'
TOKTYPE_MUL        = 'MUL'
TOKTYPE_DIV        = 'DIV'
TOKTYPE_POWER      = 'POWER'
TOKTYPE_SQRT       = 'SQRT'

# Assignment and Variables
TOKTYPE_EQUALS     = 'EQUALS'
TOKTYPE_IDENTIFIER = 'IDENTIFIER'
TOKTYPE_NUMBER     = 'NUMBER'
TOKTYPE_STRING     = 'STRING'
TOKTYPE_INT 	   = 'INT'
TOKTYPE_FLOAT      = 'FLOAT'
TOKTYPE_EOF        = 'END_OF_FILE'

# Keywords
TOKTYPE_DISPLAY    = 'DISPLAY'
TOKTYPE_WHEN       = 'WHEN'
TOKTYPE_OTHERWISE_WHEN  = 'OTHERWISE_WHEN'
TOKTYPE_IN_ANY_OTHER_CASE = "IN_ANY_OTHER_CASE"
TOKTYPE_THEN       = "THEN"
TOKTYPE_REPEAT     = 'REPEAT'
TOKTYPE_CREATE     = 'CREATE'
TOKTYPE_GIVE       = 'GIVE'
TOKTYPE_ASK        = 'ASK'
TOKTYPE_SET        = 'SET'
TOKTYPE_TO         = 'TO'
TOKTYPE_REPEAT     = "REPEAT"
TOKTYPE_FROM       = "FROM"
TOKTYPE_TO         = "TO"
TOKTYPE_STEP       = "STEP"
TOKTYPE_WHILE      = "WHILE"
TOKTYPE_TIMES       = 'TIMES'

# Symbols
TOKTYPE_LBRACKET   = 'LBRACKET'
TOKTYPE_RBRACKET   = 'RBRACKET'
TOKTYPE_COMMA      = 'COMMA'
TOKTYPE_COLON      = 'COLON'
TOKTYPE_LPAREN     = 'LEFT_PAREN'    
TOKTYPE_RPAREN     = 'RIGHT_PAREN'

# Comparison Operators
TOKTYPE_LT         = 'LESS_THAN'
TOKTYPE_GT         = 'GREATER_THAN'
TOKTYPE_LTE        = 'LESS_THAN_EQUAL'
TOKTYPE_GTE        = 'GREATER_THAN_EQUAL'
TOKTYPE_EQ         = 'EQUALS_EQUAL'
TOKTYPE_NE         = 'NOT_EQUAL'
TOKTYPE_AND        = 'AND'
TOKTYPE_OR         = 'OR'
TOKTYPE_NOT        = 'NOT'

class Token:
    def __init__(self, type_, value = None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end =pos_start.copy()
            

        if pos_end:
            self.pos_end = pos_end

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


# LEXER

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.character_current = None
        self.next_character()

    def next_character(self):
        self.pos.next_character(self.character_current)
        self.character_current = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_identifier_or_keyword(self):
        pos_start = self.pos.copy()
        identifier_str = ''
        
        while self.character_current and (self.character_current in LETTERS or self.character_current in DIGITS):
            identifier_str += self.character_current
            self.next_character()

        if identifier_str == "otherwise":
            if self.character_current and self.character_current in ' \t':
                self.next_character()
                second_word = ''
                while self.character_current and self.character_current in LETTERS:
                    second_word += self.character_current
                    self.next_character()
                if second_word == "when":
                    return Token(TOKTYPE_OTHERWISE_WHEN, pos_start=pos_start, pos_end=self.pos.copy())
                else:
                    return Token(TOKTYPE_IDENTIFIER, identifier_str + ' ' + second_word, pos_start, self.pos.copy())
        
        if identifier_str == "in":
            if self.character_current and self.character_current in ' \t':  
                self.next_character()
                second_word = ''
                while self.character_current and self.character_current in LETTERS:
                    second_word += self.character_current
                    self.next_character()
                if second_word == "any":
                    if self.character_current and self.character_current in ' \t':  
                        self.next_character()
                        third_word = ''
                        while self.character_current and self.character_current in LETTERS:
                            third_word += self.character_current
                            self.next_character()
                        if third_word == "other":
                            if self.character_current and self.character_current in ' \t':  
                                self.next_character()
                                fourth_word = ''
                                while self.character_current and self.character_current in LETTERS:
                                    fourth_word += self.character_current
                                    self.next_character()
                                if fourth_word == "case":
                                    return Token(TOKTYPE_IN_ANY_OTHER_CASE, pos_start=pos_start, pos_end=self.pos.copy())
                                
        token_type = {
            'display': TOKTYPE_DISPLAY,
            'when': TOKTYPE_WHEN,
            'repeat': TOKTYPE_REPEAT,
            'create': TOKTYPE_CREATE,
            'give': TOKTYPE_GIVE,
            'ask': TOKTYPE_ASK,
            'set': TOKTYPE_SET,
            'to': TOKTYPE_TO,
            'then': TOKTYPE_THEN,  
            'and': TOKTYPE_AND,
            'or': TOKTYPE_OR,
            'not': TOKTYPE_NOT,
            'from': TOKTYPE_FROM,
            'step': TOKTYPE_STEP,
            'while': TOKTYPE_WHILE,
            'times': TOKTYPE_TIMES
        }.get(identifier_str, TOKTYPE_IDENTIFIER)

        return Token(token_type, identifier_str if token_type == TOKTYPE_IDENTIFIER else None, pos_start, self.pos.copy())


    def make_string(self):
        pos_start = self.pos.copy()
        string_value = ''
        self.next_character()  
        escape_characters = {'n': '\n', 't': '\t', '"': '"', '\\': '\\'}

        while self.character_current:
            if self.character_current == '"':  
                break
            if self.character_current == '\\':  
                self.next_character()
                if self.character_current in escape_characters:
                    string_value += escape_characters[self.character_current]
                else:
                    string_value += '\\' + self.character_current
            else:
                string_value += self.character_current
            self.next_character()

        if self.character_current != '"':  
            return [], CharacterFormatError(pos_start, self.pos, "Unterminated string")
        self.next_character() 
        return Token(TOKTYPE_STRING, string_value, pos_start, self.pos.copy())


    def make_tokens(self):
        tokens = []
    
        SINGLE_CHAR_TOKENS = {
            '+': TOKTYPE_PLUS,
            '-': TOKTYPE_MINUS,
            '*': TOKTYPE_MUL,
            '^': TOKTYPE_POWER,
            '(': TOKTYPE_LPAREN,
            ')': TOKTYPE_RPAREN,
            '[': TOKTYPE_LBRACKET,
            ']': TOKTYPE_RBRACKET,
            ':': TOKTYPE_COLON,
            ',': TOKTYPE_COMMA,
        }
        
        while self.character_current:
            if self.character_current in ' \t\n':
                self.next_character()
            elif self.character_current in DIGITS:
                tokens.append(self.make_number())
            elif self.character_current in LETTERS:
                tokens.append(self.make_identifier_or_keyword())
            elif self.character_current == '#':
                while self.character_current and self.character_current != '\n':
                    self.next_character()
            elif self.character_current in SINGLE_CHAR_TOKENS:
                pos_start = self.pos.copy()
                tok_type = SINGLE_CHAR_TOKENS[self.character_current]
                self.next_character()
                tokens.append(Token(tok_type, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '=':
                pos_start = self.pos.copy()
                self.next_character()
                if self.character_current == '=':
                    self.next_character()
                    tokens.append(Token(TOKTYPE_EQ, pos_start=pos_start, pos_end=self.pos.copy()))
                else:
                    tokens.append(Token(TOKTYPE_EQUALS, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '!':
                pos_start = self.pos.copy()
                self.next_character()
                if self.character_current == '=':
                    self.next_character()
                    tokens.append(Token(TOKTYPE_NE, pos_start=pos_start, pos_end=self.pos.copy()))
                else:
                    return [], CharacterFormatError(pos_start, self.pos, "'!' without '='")
            elif self.character_current == '<':
                pos_start = self.pos.copy()
                self.next_character()
                if self.character_current == '=':
                    self.next_character()
                    tokens.append(Token(TOKTYPE_LTE, pos_start=pos_start, pos_end=self.pos.copy()))
                else:
                    tokens.append(Token(TOKTYPE_LT, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '>':
                pos_start = self.pos.copy()
                self.next_character()
                if self.character_current == '=':
                    self.next_character()
                    tokens.append(Token(TOKTYPE_GTE, pos_start=pos_start, pos_end=self.pos.copy()))
                else:
                    tokens.append(Token(TOKTYPE_GT, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '"':
                tokens.append(self.make_string())
            elif self.character_current == '/':
                next_char = self.peek_next()
                if next_char == '*':
                    pos_start = self.pos.copy()
                    self.next_character()  
                    self.next_character()  
                    comment_closed = False
                    while self.character_current:
                        if self.character_current == '*' and self.peek_next() == '/':
                            self.next_character()  
                            self.next_character()  
                            comment_closed = True
                            break
                        self.next_character()
                    if not comment_closed:
                        return [], CharacterFormatError(pos_start, self.pos, "Unclosed multi-line comment")
                else:
                    pos_start = self.pos.copy()
                    self.next_character()
                    tokens.append(Token(TOKTYPE_DIV, pos_start=pos_start, pos_end=self.pos.copy()))

        return tokens, None
    
    def make_number(self):
        pos_start = self.pos.copy()
        num_str = ''
        dot_count = 0

        while self.character_current and (self.character_current in DIGITS or self.character_current == '.'):
            if self.character_current == '.':
                if dot_count == 1: 
                    break
                dot_count += 1
            num_str += self.character_current
            self.next_character()

        if dot_count == 0: 
            return Token(TOKTYPE_NUMBER, int(num_str), pos_start, self.pos.copy())
        else: 
            return Token(TOKTYPE_NUMBER, float(num_str), pos_start, self.pos.copy())

    def peek_next(self):
        peek_pos = self.pos.idx + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        else:
            return None
        
#NODES

class NodeNumber:
    def __init__(self, token):
        self.token = token
        self.pos_start = token.pos_start
        self.pos_end = token.pos_end

    def __repr__(self):
        return f"{self.token}"
    

class NodeBinaryOp:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = left_node.pos_start
        self.pos_end = right_node.pos_end

    def __repr__(self):
        return f"({self.left_node} {self.op_tok} {self.right_node})"

class NodeUnaryOp:
    def __init__(self, op_tok, node):
            self.op_tok = op_tok
            self.node = node

            self.pos_start = self.op_tok.pos_start
            self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'
    
class NodeVariable:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.pos_start = var_name_tok.pos_start
        self.pos_end = var_name_tok.pos_end

    def __repr__(self):
        return f"{self.var_name_tok.value}"

    
class NodeAssign:
    def __init__(self, var_name, value):
        self.var_name = var_name
        self.value = value
        self.pos_start = var_name.pos_start
        self.pos_end = value.pos_end

    def __repr__(self):
        return f"(Assign: {self.var_name.value} = {self.value})"
    
class NodeString:
    def __init__(self, token):
        self.token = token
        self.pos_start = token.pos_start
        self.pos_end = token.pos_end

    def __repr__(self):
        return f"NodeString({self.token.value})"
    
class NodeConditional:
    def __init__(self, cases, default_case):
        self.cases = cases  
        self.default_case = default_case

    def __repr__(self):
        return f"NodeConditional(cases={self.cases}, default_case={self.default_case})"

class NodeRepeat:
    def __init__(self, var_name, start_value, end_value, step_value, body):
        self.var_name = var_name  
        self.start_value = start_value
        self.end_value = end_value
        self.step_value = step_value
        self.body = body
        self.pos_start = start_value.pos_start
        self.pos_end = body.pos_end

    def __repr__(self):
        return f"(Repeat: from {self.start_value} to {self.end_value} step {self.step_value} do {self.body})"
    
class NodeWhile:
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"NodeWhile(condition={self.condition}, body={self.body})"
    
class NodeDisplay:
    def __init__(self, value, pos_start, pos_end):
        self.value = value
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"NodeDisplay(value={self.value}, pos_start={self.pos_start}, pos_end={self.pos_end})"
    
class NodeRepeatTimes:
    def __init__(self, times_expr, body):
        self.times_expr = times_expr
        self.body = body
        self.pos_start = times_expr.pos_start
        self.pos_end = body.pos_end

    def __repr__(self):
        return f"(Repeat {self.times_expr} times: {self.body})"
    
class NodeList:
    def __init__(self, elements, pos_start, pos_end):
        self.elements = elements
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"[{', '.join(map(str, self.elements))}]"
    
class NodeSubscript:
    def __init__(self, var_name_tok, index_expr, pos_start, pos_end):
        self.var_name_tok = var_name_tok
        self.index_expr = index_expr
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return f"{self.var_name_tok.value}[{self.index_expr}]"

#THE RESULT OF PARSE

class ResultOfParse:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ResultOfParse):
            if res.error:
                self.error = res.error
            return res.node

        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self

#PARSER

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.token_current = None
        self.next_character()
        print(f"Initializing parser with tokens: {self.tokens}")

    def next_character(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.token_current = self.tokens[self.tok_idx]
        else:
            self.token_current = None
        print(f"next_character: Current token is now {self.token_current}")
        return self.token_current
    
    def peek(self):
        peek_idx = self.tok_idx + 1
        if peek_idx < len(self.tokens):
            return self.tokens[peek_idx]
        else:
            return None

    def parse(self):
        res = self.statement()
        if not res.error and self.token_current is not None:
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start, self.token_current.pos_end,
                f"Expected end of input, found '{self.token_current.value}'"
            ))
        return res

    def statement(self):
        res = ResultOfParse()

        if self.token_current and self.token_current.type == TOKTYPE_DISPLAY:
            return self.display_statement()

        if self.token_current and self.token_current.type == TOKTYPE_WHILE:
            return self.while_statement()
        
        if self.token_current and self.token_current.type == TOKTYPE_REPEAT:
            return self.repeat_statement()

        if self.token_current and self.token_current.type == TOKTYPE_WHEN:
            return self.conditional()

        if self.token_current and self.token_current.type == TOKTYPE_IDENTIFIER:
            var_name = self.token_current
            res.register(self.next_character())

            if self.token_current and self.token_current.type == TOKTYPE_EQUALS:
                res.register(self.next_character())
                expr = res.register(self.expr())
                if res.error:
                    return res
                return res.success(NodeAssign(var_name, expr))

            return res.failure(WrongSyntaxError(
                var_name.pos_start, var_name.pos_end,
                "Expected '=' after variable name"
            ))
        
        if self.token_current and self.token_current.type == TOKTYPE_CREATE:
            return self.create_statement()

        return self.expr()

    def expr(self):
        return self.logical_or()

    def logical_or(self):
        return self.bin_op(self.logical_and, (TOKTYPE_OR,))

    def logical_and(self):
        return self.bin_op(self.logical_not, (TOKTYPE_AND,))

    def logical_not(self):
        res = ResultOfParse()
        tok = self.token_current

        if tok and tok.type == TOKTYPE_NOT:
            res.register(self.next_character())
            node = res.register(self.logical_not())
            if res.error: return res
            return res.success(NodeUnaryOp(tok, node))

        return self.comparison()

    def comparison(self):
        return self.bin_op(self.term, (TOKTYPE_LT, TOKTYPE_LTE, TOKTYPE_GT, TOKTYPE_GTE, TOKTYPE_EQ, TOKTYPE_NE))

    def term(self):
        return self.bin_op(self.factor, (TOKTYPE_PLUS, TOKTYPE_MINUS))

    def factor(self):
        return self.bin_op(self.power, (TOKTYPE_MUL, TOKTYPE_DIV))

    def power(self):
        return self.bin_op(self.unary, (TOKTYPE_POWER,))

    def unary(self):
        res = ResultOfParse()
        tok = self.token_current

        if tok and tok.type in (TOKTYPE_PLUS, TOKTYPE_MINUS):
            res.register(self.next_character())
            node = res.register(self.unary())
            if res.error: return res
            return res.success(NodeUnaryOp(tok, node))

        return self.primary()

    def primary(self):
        res = ResultOfParse()
        tok = self.token_current

        if not tok:
            return res.failure(WrongSyntaxError(
                None, None,
                "Unexpected end of input, expected a number, string, or identifier"
            ))

        if tok.type == TOKTYPE_NUMBER:
            res.register(self.next_character())
            return res.success(NodeNumber(tok))

        if tok.type == TOKTYPE_STRING:
            res.register(self.next_character())
            return res.success(NodeString(tok))

        if tok.type == TOKTYPE_IDENTIFIER:
            var_name_tok = tok  
            res.register(self.next_character()) 

            if self.token_current and self.token_current.type == TOKTYPE_LBRACKET:
                res.register(self.next_character())  
                if not self.token_current:
                    return res.failure(WrongSyntaxError(
                        var_name_tok.pos_start, var_name_tok.pos_end,
                        "Unexpected end of input after '['"
                    ))

                index_expr = res.register(self.expr())  
                if res.error:
                    return res  

                if not self.token_current or self.token_current.type != TOKTYPE_RBRACKET:
                    return res.failure(WrongSyntaxError(
                        self.token_current.pos_start if self.token_current else var_name_tok.pos_start,
                        self.token_current.pos_end if self.token_current else var_name_tok.pos_end,
                        "Expected ']' after subscript index"
                    ))

                closing_bracket = self.token_current  
                res.register(self.next_character())  

                if closing_bracket is None:
                    return res.failure(WrongSyntaxError(
                        var_name_tok.pos_start, index_expr.pos_end,
                        "Unexpected end of input after subscript"
                    ))

                return res.success(NodeSubscript(var_name_tok, index_expr, var_name_tok.pos_start, closing_bracket.pos_end))

            return res.success(NodeVariable(var_name_tok))  

        if tok.type == TOKTYPE_LPAREN:
            res.register(self.next_character())
            expr = res.register(self.expr())
            if res.error: return res
            if self.token_current and self.token_current.type == TOKTYPE_RPAREN:
                res.register(self.next_character())
                return res.success(expr)
            return res.failure(WrongSyntaxError(
                tok.pos_start, tok.pos_end,
                f"Expected ')' after the expression at position {tok.pos_end}"
            ))
        
        if tok.type == TOKTYPE_LBRACKET:
            elements = []
            pos_start = tok.pos_start.copy()
            res.register(self.next_character()) 

            if self.token_current and self.token_current.type == TOKTYPE_RBRACKET:
                res.register(self.next_character())  
                return res.success(NodeList(elements, pos_start, self.token_current.pos_end.copy()))
            
            expr = res.register(self.expr())
            if res.error: return res
            elements.append(expr)

            while self.token_current and self.token_current.type == TOKTYPE_COMMA:
                res.register(self.next_character()) 
                expr = res.register(self.expr())
                if res.error: return res
                elements.append(expr)

            if not (self.token_current and self.token_current.type == TOKTYPE_RBRACKET):
                return res.failure(WrongSyntaxError(
                    self.token_current.pos_start if self.token_current else pos_start,
                    self.token_current.pos_end if self.token_current else pos_start,
                    "Expected ']' after list elements"
                ))
            
            pos_end = self.token_current.pos_end.copy()
            res.register(self.next_character())  
            return res.success(NodeList(elements, pos_start, pos_end))


            
        return res.failure(WrongSyntaxError(
                        tok.pos_start, tok.pos_end,
                        f"Unexpected token: '{tok.value}'"
                    ))
            

    def conditional(self):
        res = ResultOfParse()
        cases = []
        default_case = None

        if self.token_current and self.token_current.type == TOKTYPE_WHEN:
            print("Parsing 'when' statement")
            res.register(self.next_character())

            condition = res.register(self.expr())  
            if res.error: return res

            if self.token_current and self.token_current.type == TOKTYPE_THEN:
                res.register(self.next_character())
                body = res.register(self.statement())  
                if res.error: return res
                cases.append((condition, body, None))  
            else:
                return res.failure(WrongSyntaxError(
                    condition.pos_start, condition.pos_end,
                    "Expected 'then' after condition in 'when' statement"
                ))

            while self.token_current and self.token_current.type == TOKTYPE_OTHERWISE_WHEN:
                print("Parsing 'otherwise when' statement")
                res.register(self.next_character())

                condition = res.register(self.expr())  
                if res.error: return res

                if self.token_current and self.token_current.type == TOKTYPE_THEN:
                    res.register(self.next_character())
                    body = res.register(self.statement())  
                    if res.error: return res
                    cases.append((condition, body, None))  
                else:
                    return res.failure(WrongSyntaxError(
                        condition.pos_start, condition.pos_end,
                        "Expected 'then' after condition in 'otherwise when' statement"
                    ))

            if self.token_current and self.token_current.type == TOKTYPE_IN_ANY_OTHER_CASE:
                print("Parsing 'in any other case' statement")
                res.register(self.next_character())
                default_case = res.register(self.statement())
                if res.error: return res

            return res.success(NodeConditional(cases, default_case))

        return res.failure(WrongSyntaxError(
            self.token_current.pos_start if self.token_current else None,
            self.token_current.pos_end if self.token_current else None,
            "Expected 'when', 'otherwise when', or 'in any other case'"
        ))

    def repeat_statement(self):
        res = ResultOfParse()
        pos_start = self.token_current.pos_start.copy()

        res.register(self.next_character())

        if self.token_current and self.token_current.type == TOKTYPE_IDENTIFIER:
            var_name_tok = self.token_current
            saved_idx = self.tok_idx
            next_tok = self.peek()
            
            print(f"DEBUG: Found identifier {var_name_tok}, next token = {next_tok}")

            if next_tok and next_tok.type == TOKTYPE_FROM:
                res.register(self.next_character())  
                res.register(self.next_character())  

                start_expr = res.register(self.expr())
                if res.error: return res

                if not (self.token_current and self.token_current.type == TOKTYPE_TO):
                    return res.failure(WrongSyntaxError(
                        self.token_current.pos_start if self.token_current else pos_start,
                        self.token_current.pos_end if self.token_current else pos_start,
                        "Expected 'to' after start value"
                    ))
                res.register(self.next_character())

                end_expr = res.register(self.expr())
                if res.error: return res

                step_expr = NodeNumber(Token(TOKTYPE_NUMBER, 1, pos_start=pos_start, pos_end=pos_start))
                if self.token_current and self.token_current.type == TOKTYPE_STEP:
                    res.register(self.next_character())
                    step_expr = res.register(self.expr())
                    if res.error: return res

                if not (self.token_current and self.token_current.type == TOKTYPE_COLON):
                    return res.failure(WrongSyntaxError(
                        self.token_current.pos_start if self.token_current else pos_start,
                        self.token_current.pos_end if self.token_current else pos_start,
                        "Expected ':' after loop parameters"
                    ))
                res.register(self.next_character())

                body = res.register(self.statement())
                if res.error: return res

                return res.success(NodeRepeat(
                    var_name_tok, 
                    start_expr,
                    end_expr,
                    step_expr,
                    body
                ))

        times_expr = res.register(self.expr())
        if res.error: return res

        if not (self.token_current and self.token_current.type == TOKTYPE_TIMES):
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start if self.token_current else pos_start,
                self.token_current.pos_end if self.token_current else pos_start,
                "Expected 'times' after repetition count"
            ))
        res.register(self.next_character())

        if not (self.token_current and self.token_current.type == TOKTYPE_COLON):
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start,
                self.token_current.pos_end,
                "Expected ':' after 'times'"
            ))
        res.register(self.next_character())

        body = res.register(self.statement())
        if res.error: return res

        return res.success(NodeRepeatTimes(times_expr, body))
    
    def while_statement(self):
        res = ResultOfParse()

        if not (self.token_current and self.token_current.type == TOKTYPE_WHILE):
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start, self.token_current.pos_end,
                "Expected 'while'"
            ))

        res.register(self.next_character())

        condition = res.register(self.expr())
        if res.error: return res

        if not (self.token_current and self.token_current.type == TOKTYPE_THEN):
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start, self.token_current.pos_end,
                "Expected 'then' after while condition"
            ))

        res.register(self.next_character())
        body = res.register(self.statement())
        if res.error: return res

        return res.success(NodeWhile(condition, body))
    
    def display_statement(self):
        res = ResultOfParse()
        pos_start = self.token_current.pos_start.copy()

        if not (self.token_current and self.token_current.type == TOKTYPE_DISPLAY):
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start, self.token_current.pos_end,
                "Expected 'display'"
            ))

        res.register(self.next_character())

        if not (self.token_current and self.token_current.type == TOKTYPE_COLON):
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start, self.token_current.pos_end if self.token_current else pos_start,
                "Expected ':' after 'display'"
            ))

        res.register(self.next_character())

        value = res.register(self.expr())
        if res.error:
            return res

        pos_end = value.pos_end if hasattr(value, 'pos_end') else pos_start
        return res.success(NodeDisplay(value, pos_start, pos_end))
    
    def create_statement(self):
        res = ResultOfParse()
        create_tok = self.token_current
        res.register(self.next_character())
        if self.token_current.type != TOKTYPE_IDENTIFIER:
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start, self.token_current.pos_end,
                "Expected identifier after 'create'"
            ))
        var_name = self.token_current
        res.register(self.next_character())
        if self.token_current.type != TOKTYPE_EQUALS:
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start, self.token_current.pos_end,
                "Expected '=' after identifier in 'create'"
            ))
        res.register(self.next_character())
        expr = res.register(self.expr())
        if res.error: return res
        return res.success(NodeAssign(var_name, expr))


    def bin_op(self, func_a, ops, func_b=None):
        if func_b is None:
            func_b = func_a

        res = ResultOfParse()
        left = res.register(func_a())
        if res.error:
            return res

        while self.token_current is not None and self.token_current.type in ops:
            op_tok = self.token_current
            print(f"bin_op: Found operator '{op_tok.type}'")
            res.register(self.next_character())
            right = res.register(func_b())
            if res.error:
                return res
            left = NodeBinaryOp(left, op_tok, right)

        return res.success(left)


#RESULT OF RUNTIME

class RuntimeResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value
    

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self

#VALUES
class Number:
    def __init__(self, value: float, pos_start=None, pos_end=None, context=None):
        self.value = value
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.context = context

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def operate(self, other, op, error_message="Invalid operation"):
        if not isinstance(other, Number):
            raise ValueError(error_message)

        if op == "add":
            return Number(self.value + other.value, self.pos_start, other.pos_end, self.context), None
        elif op == "sub":
            return Number(self.value - other.value, self.pos_start, other.pos_end, self.context), None
        elif op == "mul":
            return Number(self.value * other.value, self.pos_start, other.pos_end, self.context), None
            print(f"Token generated: {tokens[-1]}")
        elif op == "div":
            if other.value == 0:
                return None, RuntimeError(
                    other.pos_start, other.pos_end, "Bruh.. You can't divide by 0...", self.context
                )
            return Number(self.value / other.value, self.pos_start, other.pos_end, self.context), None
        elif op == "pow":
            return Number(self.value ** other.value, self.pos_start, other.pos_end, self.context), None
        else:
            raise ValueError("Unsupported operation :(")

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value, self.pos_start, other.pos_end, self.context), None
        else:
            return None, TypeError(f"Cannot add Number with {type(other).__name__}")

    def subbed_by(self, other):
        return self.operate(other, "sub")

    def multed_by(self, other):
        return self.operate(other, "mul")

    def dived_by(self, other):
        return self.operate(other, "div")
    
    def powed_by(self, other):
        return self.operate(other, "pow")

    def __repr__(self):
        return str(self.value)
    
class String:
    def __init__(self, value, pos_start=None, pos_end=None, context=None):
        self.value = value
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.context = context

    def set_context(self, context):
        self.context = context
        return self

    def set_pos(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def added_to(self, other):
        if isinstance(other, String):
            return String(self.value + other.value, self.pos_start, other.pos_end, self.context), None
        elif isinstance(other, Number):
            return String(self.value + str(other.value), self.pos_start, other.pos_end, self.context), None
        else:
            return None, TypeError(f"Cannot concatenate String with {type(other).__name__}")

    def __repr__(self):
        return f'"{self.value}"'

#LIST

class List:
    def __init__(self, elements, pos_start=None, pos_end=None, context=None):
        self.elements = elements
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.context = context

    def set_context(self, context):
        self.context = context
        return self

    def set_pos(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def __repr__(self):
        return f'[{", ".join(map(str, self.elements))}]'
    
#CONTEXT

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = {}

    def set(self, var_name, value):
        self.symbol_table[var_name] = value

    def get(self, var_name):
        return self.symbol_table.get(var_name)


#INTERPRETER

class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NodeNumber(self, node, context):
        return RuntimeResult().success(
            Number(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_NodeBinaryOp(self, node, context):
        res = RuntimeResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error:
            return res

        result = None
        error = None

        if node.op_tok.type == TOKTYPE_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TOKTYPE_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TOKTYPE_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TOKTYPE_DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TOKTYPE_POWER:
            result, error = left.powed_by(right)

        elif node.op_tok.type == TOKTYPE_GT:
            result = Number(left.value > right.value)
        elif node.op_tok.type == TOKTYPE_LT: 
            result = Number(left.value < right.value)
        elif node.op_tok.type == TOKTYPE_GTE:  
            result = Number(left.value >= right.value)
        elif node.op_tok.type == TOKTYPE_LTE:  
            result = Number(left.value <= right.value)
        elif node.op_tok.type == TOKTYPE_EQ:  
            result = Number(left.value == right.value)
        elif node.op_tok.type == TOKTYPE_NE:  
            result = Number(left.value != right.value)

        elif node.op_tok.type == TOKTYPE_AND:  
            result = Number(1 if left.value and right.value else 0)
        elif node.op_tok.type == TOKTYPE_OR:  
            result = Number(1 if left.value or right.value else 0)

        else:
            return res.failure(RuntimeError(
                node.op_tok.pos_start, node.op_tok.pos_end,
                "Unknown binary operator"
            ))

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))



    def visit_NodeUnaryOp(self, node, context):
        res = RuntimeResult()
        number = res.register(self.visit(node.node, context))
        if res.error:
            return res

        error = None

        if node.op_tok.type == TOKTYPE_MINUS:
            number, error = number.multed_by(Number(-1))
        elif node.op_tok.type == TOKTYPE_SQRT:
            if number.value < 0:
                return res.failure(RuntimeError(
                    node.op_tok.pos_start, node.op_tok.pos_end,
                    "Cannot take square root of a negative number"
                ))
            number = Number(number.value ** 0.5).set_context(number.context).set_pos(node.pos_start, node.pos_end)
        elif node.op_tok.type == TOKTYPE_NOT:
            number = Number(1 if not number.value else 0).set_context(number.context).set_pos(node.pos_start, node.pos_end)

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

        
    def visit_NodeAssign(self, node, context):
        res = RuntimeResult()

        value = res.register(self.visit(node.value, context))
        if res.error:
            return res

        var_name = node.var_name.value
        context.symbol_table[var_name] = value

        return res.success(value)
    
    def visit_NodeString(self, node, context):
        return RuntimeResult().success(
            String(node.token.value, node.pos_start, node.pos_end, context)
        )

    def visit_NodeVariable(self, node, context):
        res = RuntimeResult()
        
        var_name = node.var_name_tok.value
        if var_name in context.symbol_table:
            value = context.symbol_table[var_name]
            return res.success(value)
        else:
            return res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                f"Variable '{var_name}' is not defined",
                context
            ))
        
    def visit_NodeConditional(self, node, context):
        res = RuntimeResult()

        for condition, body, _ in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error:
                return res
            
            if condition_value.value:  
                body_value = res.register(self.visit(body, context))
                if res.error:
                    return res
                return res.success(body_value)

        if node.default_case:
            default_value = res.register(self.visit(node.default_case, context))
            if res.error:
                return res
            return res.success(default_value)

        return res.success(None)

    def visit_NodeRepeat(self, node, context):
        res = RuntimeResult()

        start = res.register(self.visit(node.start_value, context))
        if res.error: return res
        end = res.register(self.visit(node.end_value, context))
        if res.error: return res
        step = res.register(self.visit(node.step_value, context))
        if res.error: return res

        if not all(isinstance(val, Number) for val in (start, end, step)):
            return res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                "Start, end, and step must be numbers",
                context
            ))
        if step.value == 0:
            return res.failure(RuntimeError(
                step.pos_start, step.pos_end,
                "Step cannot be zero",
                context
            ))

        i = start.value
        while (i <= end.value) if (step.value > 0) else (i >= end.value):
            loop_context = Context(node.var_name.value, parent=context)  
            loop_context.set(node.var_name.value, Number(i).set_context(loop_context)) 
            
            res.register(self.visit(node.body, loop_context))
            if res.error:
                return res

            i += step.value

        return res.success(None)
    
    def visit_NodeWhile(self, node, context):
        res = RuntimeResult()

        while True:
            condition = res.register(self.visit(node.condition, context))
            if res.error: 
                return res

            if not condition.value: 
                break

            res.register(self.visit(node.body, context))
            if res.error: 
                return res

        return res.success(None)
    
    def visit_NodeDisplay(self, node, context):
        res = RuntimeResult()
        value = res.register(self.visit(node.value, context))
        if res.error: 
            return res
        if isinstance(value, String):
            print(value.value)  
        else:
            print(value)
        return res.success(None)
    
    def visit_NodeRepeatTimes(self, node, context):
        res = RuntimeResult()
        times_value = res.register(self.visit(node.times_expr, context))
        if res.error:
            return res

        if not isinstance(times_value, Number):
            return res.failure(RuntimeError(
                node.times_expr.pos_start, node.times_expr.pos_end,
                "Repetition count must be a number",
                context
            ))

        try:
            times = int(times_value.value)
            if times < 0:
                return res.failure(RuntimeError(
                    node.times_expr.pos_start, node.times_expr.pos_end,
                    "Repetition count cannot be negative",
                    context
                ))
        except ValueError:
            return res.failure(RuntimeError(
                node.times_expr.pos_start, node.times_expr.pos_end,
                "Repetition count must be an integer",
                context
            ))

        for _ in range(times):
            res.register(self.visit(node.body, context))
            if res.error:
                return res

        return res.success(None)
    
    def visit_NodeList(self, node, context):
        res = RuntimeResult()
        elements = []
        for element_node in node.elements:
            element = res.register(self.visit(element_node, context))
            if res.error: return res
            elements.append(element)
        return res.success(List(elements, node.pos_start, node.pos_end, context))
    
    def visit_NodeSubscript(self, node, context):
        res = RuntimeResult()
        var_name = node.var_name_tok.value
        list_val = context.get(var_name)
        if not list_val:
            return res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                f"Undefined variable '{var_name}'", context
            ))
        if not isinstance(list_val, List):
            return res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not a list", context
            ))
        index = res.register(self.visit(node.index_expr, context))
        if res.error: return res
        if not isinstance(index, Number) or not index.value.is_integer():
            return res.failure(RuntimeError(
                node.index_expr.pos_start, node.index_expr.pos_end,
                "List index must be an integer", context
            ))
        idx = int(index.value)
        if idx < 0 or idx >= len(list_val.elements):
            return res.failure(RuntimeError(
                node.index_expr.pos_start, node.index_expr.pos_end,
                f"Index {idx} out of range", context
            ))
        return res.success(list_val.elements[idx])

#RUN

global_context = Context('<program>')

def run(fn, text, context=global_context):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()

    if error:
        return None, error
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error:
        return None, ast.error

    interpreter = Interpreter()
    result = interpreter.visit(ast.node, context)

    return result.value, result.error