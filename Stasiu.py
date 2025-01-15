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
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
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
TOKTYPE_OTHERWISE  = 'OTHERWISE'
TOKTYPE_REPEAT     = 'REPEAT'
TOKTYPE_CREATE     = 'CREATE'
TOKTYPE_GIVE       = 'GIVE'
TOKTYPE_ASK        = 'ASK'
TOKTYPE_END        = 'END'
TOKTYPE_START      = 'START'
TOKTYPE_SET        = 'SET'
TOKTYPE_TO         = 'TO'

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

        token_type = {
            'display': TOKTYPE_DISPLAY,
            'when': TOKTYPE_WHEN,
            'otherwise': TOKTYPE_OTHERWISE,
            'repeat': TOKTYPE_REPEAT,
            'create': TOKTYPE_CREATE,
            'give': TOKTYPE_GIVE,
            'ask': TOKTYPE_ASK,
            'start': TOKTYPE_START,
            'end': TOKTYPE_END,
            'set': TOKTYPE_SET,
            'to': TOKTYPE_TO,
            'sqrt': TOKTYPE_SQRT,
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
            elif self.character_current == '+':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_PLUS, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '-':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_MINUS, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '*':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_MUL, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '/':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_DIV, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '^':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_POWER, pos_start=pos_start, pos_end=self.pos.copy()))

            elif self.character_current == '=':
                pos_start = self.pos.copy()
                self.next_character()
                if self.character_current == '=': 
                    self.next_character()
                    tokens.append(Token(TOKTYPE_EQ, pos_start=pos_start, pos_end=self.pos.copy()))  
                else:  
                    tokens.append(Token(TOKTYPE_EQUALS, pos_start=pos_start, pos_end=self.pos.copy()))  
  

            elif self.character_current == '(':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_LPAREN, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == ')':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_RPAREN, pos_start=pos_start, pos_end=self.pos.copy()))
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
            elif self.character_current == '[':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_LBRACKET, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == ']':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_RBRACKET, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == ':':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_COLON, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == ',':
                pos_start = self.pos.copy()
                self.next_character()
                tokens.append(Token(TOKTYPE_COMMA, pos_start=pos_start, pos_end=self.pos.copy()))
            elif self.character_current == '"':
                tokens.append(self.make_string())
            else:
                pos_start = self.pos.copy()
                char = self.character_current
                self.next_character()
                return [], CharacterFormatError(pos_start, self.pos, "'" + char + "'")

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
        return f"(String: {self.token.value})"



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


    def parse(self):
        res = self.statement()  
        if not res.error and self.token_current is not None:
            return res.failure(WrongSyntaxError(
                self.token_current.pos_start, self.token_current.pos_end,
                "Expected end of input"
            ))
        return res


    def factor(self):
        print(f"Parsing factor with current token: {self.token_current}")
        res = ResultOfParse()
        tok = self.token_current

        if not tok:
            return res.failure(WrongSyntaxError(
                self.tokens[self.tok_idx - 1].pos_end if self.tok_idx > 0 else None,
                None,
                "Unexpected end of input, expected a number, string, or '('"
            ))

        if tok.type in (TOKTYPE_PLUS, TOKTYPE_MINUS):  
            res.register(self.next_character())
            factor = res.register(self.factor())
            if res.error: 
                return res
            return res.success(NodeUnaryOp(tok, factor))

        elif tok.type == TOKTYPE_NUMBER:  
            res.register(self.next_character())
            return res.success(NodeNumber(tok))

        elif tok.type == TOKTYPE_STRING:  
            res.register(self.next_character())
            return res.success(NodeString(tok))  

        elif tok.type == TOKTYPE_IDENTIFIER:  
            res.register(self.next_character())
            return res.success(NodeVariable(tok))  

        elif tok.type == TOKTYPE_LPAREN:  
            res.register(self.next_character())
            expr = res.register(self.expr())  
            if res.error:
                return res
            if self.token_current and self.token_current.type == TOKTYPE_RPAREN:  
                res.register(self.next_character())
                return res.success(expr)
            else:
                return res.failure(WrongSyntaxError(
                    tok.pos_start, tok.pos_end,
                    f"Expected ')' after the expression at position {tok.pos_end}"
                ))

        return res.failure(WrongSyntaxError(
            tok.pos_start, tok.pos_end,
            f"Expected a number, string, '(', or identifier but found '{tok.value}'"
        ))


    def statement(self):
        if self.token_current and self.token_current.type == TOKTYPE_IDENTIFIER:
            print(f"statement: Found identifier {self.token_current.value}")
            var_name = self.token_current
            self.next_character()
            print(f"statement: Next token after identifier: {self.token_current}")
            if self.token_current and self.token_current.type == TOKTYPE_EQUALS:  
                print(f"statement: Found '=' after identifier {var_name.value}")
                self.next_character()
                res = self.expr()  
                if res.error:
                    return res  
                expr = res.node  
                return ResultOfParse().success(NodeAssign(var_name, expr))  
            else:
                return ResultOfParse().failure(WrongSyntaxError(
                    var_name.pos_start, var_name.pos_end,
                    "Expected '=' after variable name"
                ))
        return self.expr()  





    def power(self):  
        return self.bin_op(self.factor, (TOKTYPE_POWER,))
    
    def term(self):
        return self.bin_op(self.power, (TOKTYPE_MUL, TOKTYPE_DIV))


    def expr(self):
        return self.bin_op(self.term, (TOKTYPE_PLUS, TOKTYPE_MINUS))


    def bin_op(self, func, ops):
        res = ResultOfParse()
        left = res.register(func())
        if res.error:
            return res

        print(f"bin_op: Entering loop with current token {self.token_current} and valid ops {ops}")
        while self.token_current is not None and self.token_current.type in ops:
            print(f"bin_op: Current token is {self.token_current}, valid ops are {ops}")
            op_tok = self.token_current
            res.register(self.next_character())
            print(f"bin_op: After next_character, current token: {self.token_current}")

            if not self.token_current:
                return res.failure(WrongSyntaxError(
                    op_tok.pos_start, op_tok.pos_end,
                    f"Expected a value after operator '{op_tok.value}'"
                ))

            right = res.register(func())
            print(f"bin_op: Parsed right operand: {right}")
            if res.error:
                return res
            left = NodeBinaryOp(left, op_tok, right)
            print(f"bin_op: Created NodeBinaryOp: {left}")

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
        return self.operate(other, "add")

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
    def __init__(self, value):
        self.value = value
        self.pos_start = None
        self.pos_end = None
        self.context = None

    def set_context(self, context):
        self.context = context
        return self

    def set_pos(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def __repr__(self):
        return f'"{self.value}"'

    
#CONTEXT

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = {}


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
            String(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )



#RUN

def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error
    
    interpreter = Interpreter()
    context = Context('<program>')
    result = interpreter.visit(ast.node, context)

    return result.value, result.error