#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

const int max_size = 50;

// 栈节点结构体
typedef struct Stacknode {
    void* data;  
    struct Stacknode* next;
} Stacknode;

// 栈结构体
typedef struct Stack {
    Stacknode* top;
} Stack;

// 初始化栈
Stack* init_stack() {
    Stack* stack = (Stack*)malloc(sizeof(Stack));
    stack->top = NULL;
    return stack;
}

// 入栈（支持任意类型数据）
void push_stack(Stack* stack, void* data, size_t size) {
    Stacknode* newnode = (Stacknode*)malloc(sizeof(Stacknode));
    if (newnode) {
        newnode->data = malloc(size);  // 根据传入的 size 分配内存
        if (newnode->data) {
            memcpy(newnode->data, data, size);  // 复制数据
            newnode->next = stack->top;
            stack->top = newnode;
        } else {
            printf("malloc failed for data\n");
        }
    } else {
        printf("malloc failed for newnode\n");
    }
}

// 出栈（返回数据指针，调用者需负责释放数据内存）
void* pop_stack(Stack* stack) {
    void* data = NULL;
    if (stack->top) {
        Stacknode* p = stack->top;
        data = stack->top->data;  // 返回数据指针
        stack->top = stack->top->next;
        free(p);  // 释放 Stacknode
    } else {
        printf("stack is empty\n");
    }
    return data;
}

// 获取栈顶元素（不弹出）
void* peek_stack(Stack* stack) {
    if (stack->top) {
        return stack->top->data;
    }
    return NULL;
}

// 判断是否为运算符
int is_operator(const char* token) {
    return strcmp(token, "+") == 0 || strcmp(token, "-") == 0 ||
           strcmp(token, "*") == 0 || strcmp(token, "/") == 0;
}

// 获取运算符的优先级
int precedence(const char* op) {
    if (strcmp(op, "+") == 0 || strcmp(op, "-") == 0) {
        return 1;
    } else if (strcmp(op, "*") == 0 || strcmp(op, "/") == 0) {
        return 2;
    }
    return 0;
}

// 将中缀表达式转换为后缀表达式
char** turn_to_suffix(char* str[], int len) {
    Stack* stack = init_stack();
    char** output = (char**)malloc(max_size * sizeof(char*));
    int output_index = 0;

    for (int i = 0; str[i] ; i++) {
        char* token = str[i];

        if (isdigit(token[0]) || (token[0] == '-' && isdigit(token[1]))) {
            // 如果是数字（包括负数和小数），直接输出
            output[output_index++] = strdup(token);
        } else if (is_operator(token)) {
            // 如果是运算符，弹出栈中优先级大于等于当前运算符的运算符
            while (stack->top != NULL && is_operator((char*)peek_stack(stack)) &&
                   precedence((char*)peek_stack(stack)) >= precedence(token)) {
                output[output_index++] = strdup((char*)pop_stack(stack));
            }
            push_stack(stack, token, sizeof(char*));
        } else if (strcmp(token, "(") == 0) {
            // 如果是左括号，直接入栈
            push_stack(stack, token, sizeof(char*));
        } else if (strcmp(token, ")") == 0) {
            // 如果是右括号，弹出栈中元素直到遇到左括号
            while (stack->top != NULL && strcmp((char*)peek_stack(stack), "(") != 0) {
                output[output_index++] = strdup((char*)pop_stack(stack));
            }
            if (stack->top != NULL && strcmp((char*)peek_stack(stack), "(") == 0) {
                pop_stack(stack);  // 弹出左括号
            }
        }
    }

    // 弹出栈中剩余的运算符
    while (stack->top != NULL) {
        output[output_index++] = strdup((char*)pop_stack(stack));
    }

    // 添加结束标记
    output[output_index] = NULL;

    // 释放栈
    while (stack->top != NULL) {
        free(pop_stack(stack));
    }
    free(stack);

    return output;
}

// 计算后缀表达式的值
double calculate(char** str) {
    Stack* stack = init_stack();

    for (int i = 0; str[i] != NULL; i++) {
        char* token = str[i];

        if (isdigit(token[0]) || (token[0] == '-' && isdigit(token[1]))) {
            // 如果是数字（包括负数和小数），将其转换为 double 并压入栈中
            double num = atof(token);
            push_stack(stack, &num, sizeof(double));
        } else if (is_operator(token)) {
            // 如果是运算符，弹出两个操作数进行计算
            double* b = (double*)pop_stack(stack);
            double* a = (double*)pop_stack(stack);
            if (a == NULL || b == NULL) {
                printf("Invalid expression\n");
                return 0.0;
            }

            double result = 0.0;
            if (strcmp(token, "+") == 0) {
                result = *a + *b;
            } else if (strcmp(token, "-") == 0) {
                result = *a - *b;
            } else if (strcmp(token, "*") == 0) {
                result = *a * *b;
            } else if (strcmp(token, "/") == 0) {
                if (*b == 0.0) {
                    printf("Division by zero\n");
                    return 0.0;
                }
                result = *a / *b;
            }

            // 将计算结果压入栈中
            push_stack(stack, &result, sizeof(double));

            // 释放内存
            free(a);
            free(b);
        }
    }

    // 最终栈中剩下结果
    double* result = (double*)pop_stack(stack);
    if (result == NULL) {
        printf("Invalid expression\n");
        return 0.0;
    }

    double final_result = *result;
    free(result);

    // 释放栈
    while (stack->top != NULL) {
        free(pop_stack(stack));
    }
    free(stack);

    return final_result;
}


//==========================================================================================================================
//=====================================================================================================================

// 判断字符是否为运算符或括号
int is_operator_or_bracket(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '(' || c == ')';
}

// 将输入的字符串转换为指针数组
char** due_input(char* str) {
    char** result = (char**)malloc(max_size * sizeof(char*));
    int result_index = 0;

    int i = 0;
    while (str[i] != '\0') {
        // 跳过空格
        if (isspace(str[i])) {
            i++;
            continue;
        }

        // 处理负数或减号
        if (str[i] == '-') {
            if (i == 0 || is_operator_or_bracket(str[i - 1]) || str[i - 1] == '(') {
                // 如果是负数，读取整个数字
                int j = i + 1;
                while (isdigit(str[j]) || str[j] == '.') {
                    j++;
                }
                int len = j - i;
                result[result_index] = (char*)malloc((len + 1) * sizeof(char));
                strncpy(result[result_index], str + i, len);
                result[result_index][len] = '\0';
                result_index++;
                i = j;
                continue;
            } else {
                // 如果是减号，单独作为一个运算符
                result[result_index] = (char*)malloc(2 * sizeof(char));
                result[result_index][0] = str[i];
                result[result_index][1] = '\0';
                result_index++;
                i++;
                continue;
            }
        }

        // 处理数字（包括小数）
        if (isdigit(str[i]) || str[i] == '.') {
            int j = i;
            while (isdigit(str[j]) || str[j] == '.') {
                j++;
            }
            int len = j - i;
            result[result_index] = (char*)malloc((len + 1) * sizeof(char));
            strncpy(result[result_index], str + i, len);
            result[result_index][len] = '\0';
            result_index++;
            i = j;
            continue;
        }

        // 处理运算符或括号
        if (is_operator_or_bracket(str[i])) {
            result[result_index] = (char*)malloc(2 * sizeof(char));
            result[result_index][0] = str[i];
            result[result_index][1] = '\0';
            result_index++;
            i++;
            continue;
        }

        // 其他字符（非法字符）
        printf("Invalid character: %c\n", str[i]);
        i++;
    }

    // 添加结束标记
    result[result_index] = NULL;

    return result;
}

//====================================================================================================================
// 检查输入是否合法
int check_input(char *str){
    int legal = 1;     //合法
    int khnum = 0;  //括号差
    for(int i=0; str[i] != '\0'; i++){
        if(str[i+1] == '\0' || str[i+1] == '\n'){
            if(str[i] == '+' || str[i] == '-' || str[i] == '*' || str[i] == '/' || str[i] == '(' || str[i] == '.'){
                legal = 0; break;
            }
            if(str[i] == ')'){
                khnum--;
            }
        }
        else if(isspace(*str)){     //跳过空格
            continue;
        }
        else if(str[i] >= '0' && str[i] <= '9'){ //跳过数字
            continue;

        }
        else if(str[i] == '.'){  //小数点
            if(i == 0){ legal = 0; break;
            }else if(!(str[i-1] >= '0' && str[i-1] <= '9') || !(str[i+1] >= '0' && str[i+1] <= '9') ||
             str[i+1] == '.' || str[i-1] == '.'){
                printf("illegal '%c' ", str[i]);
                legal = 0; break;
            }
            continue;
        }
        else if(str[i] == '-'){      //减（负）号
            if(i != 0){
                if( str[i+1] == '+' || str[i+1] == '-' || str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ){
                        printf("illegal '%c' ", str[i]);
                    legal = 0; break;
                }
            }
            continue;
        }
        else if(str[i] == '+'){    //加号
            if(i == 0){ legal = 0; break;
            }else if(str[i-1] == '+' || str[i-1] == '-' || str[i-1] == '*' || str[i-1] == '/' || str[i-1] == '(' ||
                     str[i+1] == '+' || str[i+1] == '-' || str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ){
                legal = 0; break;
            }
            continue;
        }
        else if(str[i] == '*' || str[i] == '/'){       //乘除号
            if(i == 0){ legal = 0; break;
            }else if(str[i-1] == '+' || str[i-1] == '-' || str[i-1] == '*' || str[i-1] == '/' || str[i-1] == '(' ||
                     str[i+1] == '+' ||                    str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ){
                legal = 0; break;
            }else if(str[i] == '/' && str[i+1] == '0'){
                legal = 0; break;
            }
            continue;
        }
        else if(str[i] == '('){    //左括号
            if(i == 0){   //式首
                if(  str[i+1] == '+' || str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ){
                    legal = 0; break;
                }
            }else{      //式中
                if(str[i+1] == '+' || str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ||
                   str[i-1] == ')' ){
                    legal = 0; break;
                }
            }
            khnum++;
            continue;
        }
        else if(str[i] == ')'){    //右括号
            if(i == 0){ legal = 0; break;
            }else if(str[i-1] == '+' || str[i-1] == '-' || str[i-1] == '*' || str[i-1] == '/' ||
                     str[i+1] == '(' ){
                        legal = 0; break;
            }
            khnum--;
            continue;
        }
        else{   
            printf("非法字符 '%c'", str[i]);
            legal = 0;
            break;
        }
    }
    if(khnum != 0){     //判断括号数量
        printf("illegal kuohao\n");
        legal = 0;
    }
    return legal;
}
/*
// 测试函数     T
int main() {
    /*
    char* input[] = {"-1.0", "-", "(", "-1", "+", "1.4", ")", "*", "-2", NULL};
    int len = sizeof(input) / sizeof(input[0]);
    for(int i=0; input[i] != NULL; i++){
        printf(" '%s' ", input[i]);
    }
    len--;
    */
//add
/*
    char input[] = "-1.0+1+2";
    char** input_ = due_input(input);
    int len = sizeof(input_)/sizeof(input_[0]);
    int i=0;
    for(i=0; input_[i]; i++){
        printf("  '%s',  ", input_[i]);
    }
//
    
    // 将中缀表达式转换为后缀表达式
    char** output = turn_to_suffix(input_, len);

    // 打印后缀表达式
    printf("后缀表达式: ");
    for (int i = 0; output[i] != NULL; i++) {
        printf("%s ", output[i]);
    }
    printf("\n");

    // 计算后缀表达式的值
    double result = calculate(output);
    printf("计算结果: %f\n", result);

    // 释放输出数组的内存
    for (int i = 0; output[i] != NULL; i++) {
        free(output[i]);
    }
    free(output);

    return 0;
}
*/
/*
// 测试函数     T

int main() {
    char input[] = "-1.0+2*(-3*5)/-4-(-1)";
    char** output = due_input(input);

    printf("输入字符串: %s\n", input);
    printf("解析结果: ");
    for (int i = 0; output[i] != NULL; i++) {
        printf("\"%s\" ", output[i]);
        free(output[i]);  // 释放每个字符串的内存
    }
    printf("\n");

    free(output);  // 释放输出数组的内存

    return 0;
}
    */
   
int main(){         //Finaltest
    char input[max_size];
    printf("输入“+, -, *, /, (, )” 请勿输入中文括号, 数位不超过 max_size(输那么长干嘛), 算式中元素不应超过 max_size 个\n");
    fgets(input, max_size, stdin);
    int len = strlen(input);
    if(len>0 && input[len-1] == '\n'){
        input[len-1] = '\0';
    }
    int check = check_input(input);//检查输入合法
    
    if(check){
        char** fomular = (char**)malloc(max_size*sizeof(char[max_size]));
        //char** 
        fomular = due_input(input);
        printf("分割字符串");
        for(int i=0; fomular[i]; i++){
            printf(" '%s',  ", fomular[i]);
        }
        char** suffix = (char**)malloc(max_size*sizeof(char[max_size]));
        int len_s = sizeof(fomular)/(sizeof(fomular[0]));
        suffix = turn_to_suffix(fomular, len_s);
        //char** suffix = turn_to_suffix(fomular, len_s);
        free(fomular);
        printf("后缀表达式: ");
        for (int i = 0; suffix[i] != NULL; i++) {
            printf("%s ", suffix[i]);
        }
        printf("\n");
        double result = calculate(suffix);
        printf("result = %f", result);
        
        free(suffix);
    }else{
        printf("illegal input");
    }
}
    
   /*
int main() {            //TEST FAILED
    char fomular[max_size];
    printf("输入“+, -, *, /, (, )” 请勿输入中文括号, 数位不超过 max_size(输那么长干嘛), 算式中元素不应超过 max_size 个\n");
    
    // 使用 fgets 读取输入
    if (fgets(fomular, max_size, stdin)) {
        // 去掉换行符
        int len = strlen(fomular);
        if (len > 0 && fomular[len - 1] == '\n') {
            fomular[len - 1] = '\0';
        }

    char** input = due_input(fomular);
    // 将中缀表达式转换为后缀表达式
    char** output = turn_to_suffix(input, len);

    // 打印后缀表达式
    printf("后缀表达式: ");
    for (int i = 0; output[i] != NULL; i++) {
        printf("%s ", output[i]);
    }
    printf("\n");

    // 计算后缀表达式的值
    double result = calculate(output);
    printf("计算结果: %.2f\n", result);

    // 释放输出数组的内存
    for (int i = 0; output[i] != NULL; i++) {
        free(output[i]);
    }
    free(output);
    }else {
        printf("读取失败");
    }

    return 0;
}
    

int check_input(char *str){
return 1;
}
*/