#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

const int max_size = 50;

// ջ�ڵ�ṹ��
typedef struct Stacknode {
    void* data;  
    struct Stacknode* next;
} Stacknode;

// ջ�ṹ��
typedef struct Stack {
    Stacknode* top;
} Stack;

// ��ʼ��ջ
Stack* init_stack() {
    Stack* stack = (Stack*)malloc(sizeof(Stack));
    stack->top = NULL;
    return stack;
}

// ��ջ��֧�������������ݣ�
void push_stack(Stack* stack, void* data, size_t size) {
    Stacknode* newnode = (Stacknode*)malloc(sizeof(Stacknode));
    if (newnode) {
        newnode->data = malloc(size);  // ���ݴ���� size �����ڴ�
        if (newnode->data) {
            memcpy(newnode->data, data, size);  // ��������
            newnode->next = stack->top;
            stack->top = newnode;
        } else {
            printf("malloc failed for data\n");
        }
    } else {
        printf("malloc failed for newnode\n");
    }
}

// ��ջ����������ָ�룬�������踺���ͷ������ڴ棩
void* pop_stack(Stack* stack) {
    void* data = NULL;
    if (stack->top) {
        Stacknode* p = stack->top;
        data = stack->top->data;  // ��������ָ��
        stack->top = stack->top->next;
        free(p);  // �ͷ� Stacknode
    } else {
        printf("stack is empty\n");
    }
    return data;
}

// ��ȡջ��Ԫ�أ���������
void* peek_stack(Stack* stack) {
    if (stack->top) {
        return stack->top->data;
    }
    return NULL;
}

// �ж��Ƿ�Ϊ�����
int is_operator(const char* token) {
    return strcmp(token, "+") == 0 || strcmp(token, "-") == 0 ||
           strcmp(token, "*") == 0 || strcmp(token, "/") == 0;
}

// ��ȡ����������ȼ�
int precedence(const char* op) {
    if (strcmp(op, "+") == 0 || strcmp(op, "-") == 0) {
        return 1;
    } else if (strcmp(op, "*") == 0 || strcmp(op, "/") == 0) {
        return 2;
    }
    return 0;
}

// ����׺���ʽת��Ϊ��׺���ʽ
char** turn_to_suffix(char* str[], int len) {
    Stack* stack = init_stack();
    char** output = (char**)malloc(max_size * sizeof(char*));
    int output_index = 0;

    for (int i = 0; str[i] ; i++) {
        char* token = str[i];

        if (isdigit(token[0]) || (token[0] == '-' && isdigit(token[1]))) {
            // ��������֣�����������С������ֱ�����
            output[output_index++] = strdup(token);
        } else if (is_operator(token)) {
            // ����������������ջ�����ȼ����ڵ��ڵ�ǰ������������
            while (stack->top != NULL && is_operator((char*)peek_stack(stack)) &&
                   precedence((char*)peek_stack(stack)) >= precedence(token)) {
                output[output_index++] = strdup((char*)pop_stack(stack));
            }
            push_stack(stack, token, sizeof(char*));
        } else if (strcmp(token, "(") == 0) {
            // ����������ţ�ֱ����ջ
            push_stack(stack, token, sizeof(char*));
        } else if (strcmp(token, ")") == 0) {
            // ����������ţ�����ջ��Ԫ��ֱ������������
            while (stack->top != NULL && strcmp((char*)peek_stack(stack), "(") != 0) {
                output[output_index++] = strdup((char*)pop_stack(stack));
            }
            if (stack->top != NULL && strcmp((char*)peek_stack(stack), "(") == 0) {
                pop_stack(stack);  // ����������
            }
        }
    }

    // ����ջ��ʣ��������
    while (stack->top != NULL) {
        output[output_index++] = strdup((char*)pop_stack(stack));
    }

    // ��ӽ������
    output[output_index] = NULL;

    // �ͷ�ջ
    while (stack->top != NULL) {
        free(pop_stack(stack));
    }
    free(stack);

    return output;
}

// �����׺���ʽ��ֵ
double calculate(char** str) {
    Stack* stack = init_stack();

    for (int i = 0; str[i] != NULL; i++) {
        char* token = str[i];

        if (isdigit(token[0]) || (token[0] == '-' && isdigit(token[1]))) {
            // ��������֣�����������С����������ת��Ϊ double ��ѹ��ջ��
            double num = atof(token);
            push_stack(stack, &num, sizeof(double));
        } else if (is_operator(token)) {
            // �����������������������������м���
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

            // ��������ѹ��ջ��
            push_stack(stack, &result, sizeof(double));

            // �ͷ��ڴ�
            free(a);
            free(b);
        }
    }

    // ����ջ��ʣ�½��
    double* result = (double*)pop_stack(stack);
    if (result == NULL) {
        printf("Invalid expression\n");
        return 0.0;
    }

    double final_result = *result;
    free(result);

    // �ͷ�ջ
    while (stack->top != NULL) {
        free(pop_stack(stack));
    }
    free(stack);

    return final_result;
}


//==========================================================================================================================
//=====================================================================================================================

// �ж��ַ��Ƿ�Ϊ�����������
int is_operator_or_bracket(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '(' || c == ')';
}

// ��������ַ���ת��Ϊָ������
char** due_input(char* str) {
    char** result = (char**)malloc(max_size * sizeof(char*));
    int result_index = 0;

    int i = 0;
    while (str[i] != '\0') {
        // �����ո�
        if (isspace(str[i])) {
            i++;
            continue;
        }

        // �����������
        if (str[i] == '-') {
            if (i == 0 || is_operator_or_bracket(str[i - 1]) || str[i - 1] == '(') {
                // ����Ǹ�������ȡ��������
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
                // ����Ǽ��ţ�������Ϊһ�������
                result[result_index] = (char*)malloc(2 * sizeof(char));
                result[result_index][0] = str[i];
                result[result_index][1] = '\0';
                result_index++;
                i++;
                continue;
            }
        }

        // �������֣�����С����
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

        // ���������������
        if (is_operator_or_bracket(str[i])) {
            result[result_index] = (char*)malloc(2 * sizeof(char));
            result[result_index][0] = str[i];
            result[result_index][1] = '\0';
            result_index++;
            i++;
            continue;
        }

        // �����ַ����Ƿ��ַ���
        printf("Invalid character: %c\n", str[i]);
        i++;
    }

    // ��ӽ������
    result[result_index] = NULL;

    return result;
}

//====================================================================================================================
// ��������Ƿ�Ϸ�
int check_input(char *str){
    int legal = 1;     //�Ϸ�
    int khnum = 0;  //���Ų�
    for(int i=0; str[i] != '\0'; i++){
        if(str[i+1] == '\0' || str[i+1] == '\n'){
            if(str[i] == '+' || str[i] == '-' || str[i] == '*' || str[i] == '/' || str[i] == '(' || str[i] == '.'){
                legal = 0; break;
            }
            if(str[i] == ')'){
                khnum--;
            }
        }
        else if(isspace(*str)){     //�����ո�
            continue;
        }
        else if(str[i] >= '0' && str[i] <= '9'){ //��������
            continue;

        }
        else if(str[i] == '.'){  //С����
            if(i == 0){ legal = 0; break;
            }else if(!(str[i-1] >= '0' && str[i-1] <= '9') || !(str[i+1] >= '0' && str[i+1] <= '9') ||
             str[i+1] == '.' || str[i-1] == '.'){
                printf("illegal '%c' ", str[i]);
                legal = 0; break;
            }
            continue;
        }
        else if(str[i] == '-'){      //����������
            if(i != 0){
                if( str[i+1] == '+' || str[i+1] == '-' || str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ){
                        printf("illegal '%c' ", str[i]);
                    legal = 0; break;
                }
            }
            continue;
        }
        else if(str[i] == '+'){    //�Ӻ�
            if(i == 0){ legal = 0; break;
            }else if(str[i-1] == '+' || str[i-1] == '-' || str[i-1] == '*' || str[i-1] == '/' || str[i-1] == '(' ||
                     str[i+1] == '+' || str[i+1] == '-' || str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ){
                legal = 0; break;
            }
            continue;
        }
        else if(str[i] == '*' || str[i] == '/'){       //�˳���
            if(i == 0){ legal = 0; break;
            }else if(str[i-1] == '+' || str[i-1] == '-' || str[i-1] == '*' || str[i-1] == '/' || str[i-1] == '(' ||
                     str[i+1] == '+' ||                    str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ){
                legal = 0; break;
            }else if(str[i] == '/' && str[i+1] == '0'){
                legal = 0; break;
            }
            continue;
        }
        else if(str[i] == '('){    //������
            if(i == 0){   //ʽ��
                if(  str[i+1] == '+' || str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ){
                    legal = 0; break;
                }
            }else{      //ʽ��
                if(str[i+1] == '+' || str[i+1] == '*' || str[i+1] == '/' || str[i+1] == ')' ||
                   str[i-1] == ')' ){
                    legal = 0; break;
                }
            }
            khnum++;
            continue;
        }
        else if(str[i] == ')'){    //������
            if(i == 0){ legal = 0; break;
            }else if(str[i-1] == '+' || str[i-1] == '-' || str[i-1] == '*' || str[i-1] == '/' ||
                     str[i+1] == '(' ){
                        legal = 0; break;
            }
            khnum--;
            continue;
        }
        else{   
            printf("�Ƿ��ַ� '%c'", str[i]);
            legal = 0;
            break;
        }
    }
    if(khnum != 0){     //�ж���������
        printf("illegal kuohao\n");
        legal = 0;
    }
    return legal;
}
/*
// ���Ժ���     T
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
    
    // ����׺���ʽת��Ϊ��׺���ʽ
    char** output = turn_to_suffix(input_, len);

    // ��ӡ��׺���ʽ
    printf("��׺���ʽ: ");
    for (int i = 0; output[i] != NULL; i++) {
        printf("%s ", output[i]);
    }
    printf("\n");

    // �����׺���ʽ��ֵ
    double result = calculate(output);
    printf("������: %f\n", result);

    // �ͷ����������ڴ�
    for (int i = 0; output[i] != NULL; i++) {
        free(output[i]);
    }
    free(output);

    return 0;
}
*/
/*
// ���Ժ���     T

int main() {
    char input[] = "-1.0+2*(-3*5)/-4-(-1)";
    char** output = due_input(input);

    printf("�����ַ���: %s\n", input);
    printf("�������: ");
    for (int i = 0; output[i] != NULL; i++) {
        printf("\"%s\" ", output[i]);
        free(output[i]);  // �ͷ�ÿ���ַ������ڴ�
    }
    printf("\n");

    free(output);  // �ͷ����������ڴ�

    return 0;
}
    */
   
int main(){         //Finaltest
    char input[max_size];
    printf("���롰+, -, *, /, (, )�� ����������������, ��λ������ max_size(����ô������), ��ʽ��Ԫ�ز�Ӧ���� max_size ��\n");
    fgets(input, max_size, stdin);
    int len = strlen(input);
    if(len>0 && input[len-1] == '\n'){
        input[len-1] = '\0';
    }
    int check = check_input(input);//�������Ϸ�
    
    if(check){
        char** fomular = (char**)malloc(max_size*sizeof(char[max_size]));
        //char** 
        fomular = due_input(input);
        printf("�ָ��ַ���");
        for(int i=0; fomular[i]; i++){
            printf(" '%s',  ", fomular[i]);
        }
        char** suffix = (char**)malloc(max_size*sizeof(char[max_size]));
        int len_s = sizeof(fomular)/(sizeof(fomular[0]));
        suffix = turn_to_suffix(fomular, len_s);
        //char** suffix = turn_to_suffix(fomular, len_s);
        free(fomular);
        printf("��׺���ʽ: ");
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
    printf("���롰+, -, *, /, (, )�� ����������������, ��λ������ max_size(����ô������), ��ʽ��Ԫ�ز�Ӧ���� max_size ��\n");
    
    // ʹ�� fgets ��ȡ����
    if (fgets(fomular, max_size, stdin)) {
        // ȥ�����з�
        int len = strlen(fomular);
        if (len > 0 && fomular[len - 1] == '\n') {
            fomular[len - 1] = '\0';
        }

    char** input = due_input(fomular);
    // ����׺���ʽת��Ϊ��׺���ʽ
    char** output = turn_to_suffix(input, len);

    // ��ӡ��׺���ʽ
    printf("��׺���ʽ: ");
    for (int i = 0; output[i] != NULL; i++) {
        printf("%s ", output[i]);
    }
    printf("\n");

    // �����׺���ʽ��ֵ
    double result = calculate(output);
    printf("������: %.2f\n", result);

    // �ͷ����������ڴ�
    for (int i = 0; output[i] != NULL; i++) {
        free(output[i]);
    }
    free(output);
    }else {
        printf("��ȡʧ��");
    }

    return 0;
}
    

int check_input(char *str){
return 1;
}
*/