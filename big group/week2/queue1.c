#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct Queuenode {
    void* data;
    struct Queuenode* next;
} Queuenode;

typedef struct Queue {
    Queuenode* head;
    Queuenode* tail;
    int length;
} Queue;

Queue* init_queue() {
    Queue* queue = (Queue*)malloc(sizeof(Queue));
    if (!queue) return NULL;
    queue->head = NULL;
    queue->tail = NULL;
    queue->length = 0;
    return queue;
}

void join_queue(Queue* queue, void* data, size_t data_size) {
    if (!queue || !data) return;

    Queuenode* newnode = (Queuenode*)malloc(sizeof(Queuenode));
    if (!newnode) {
        perror("malloc failed for newnode");
        return;
    }

    newnode->data = malloc(data_size);
    if (!newnode->data) {
        free(newnode);
        perror("malloc failed for data");
        return;
    }

    memcpy(newnode->data, data, data_size);
    newnode->next = NULL;

    if (!queue->head) {
        queue->head = queue->tail = newnode;
    } else {
        queue->head->next = newnode;
        queue->head = newnode;
    }
    queue->length++;
}

void* quit_queue(Queue* queue) {
    if (!queue || !queue->tail) {
        printf("Queue is empty\n");
        return NULL;
    }

    Queuenode* temp = queue->tail;
    void* data = temp->data;
    queue->tail = temp->next;

    if (!queue->tail) {
        queue->head = NULL; // 队列已空
    }

    free(temp);
    queue->length--;
    return data;
}

void* get_queue_head(Queue* queue) {
    if (!queue || !queue->head) {
        printf("queue is empty\n");
        return NULL;
    }
    return queue->head->data;
}

void clean_queue(Queue* queue) {
    if (!queue) return;

    while (queue->tail) {
        Queuenode* temp = queue->tail;
        queue->tail = temp->next;
        free(temp->data);
        free(temp);
    }
    queue->head = NULL;
    queue->length = 0;
    printf("clean succeed\n");
}

void free_queue(Queue* queue) {
    if (!queue) return;
    clean_queue(queue);
    free(queue);
    printf("free succeed\n");
}

/*
int main() {
    Queue* queue = init_queue();
    if (!queue) return 1;

    int a1 = 114514;
    double a2 = 1919810.0;
    char a3[] = "lingganguliguli";

    // 测试空队列
    assert(get_queue_head(queue) == NULL);

    // 入队
    join_queue(queue, &a1, sizeof(a1));
    join_queue(queue, &a2, sizeof(a2));
    join_queue(queue, a3, sizeof(a3));

    // 出队
    int* b1 = (int*)quit_queue(queue);
    double* b2 = (double*)quit_queue(queue);
    char* b3 = (char*)quit_queue(queue);
    
    if (b1) printf("%d\n", *b1);
    if (b2) printf("%f\n", *b2);
    if (b3) printf("%s\n", b3);

    // 释放内存
    free(b1);
    free(b2);
    free(b3);

    // 重新入队测试
    join_queue(queue, &a1, sizeof(a1));
    int* temp = (int*)get_queue_head(queue);
    if (temp) printf("temp: %d\n", *temp);

    free_queue(queue);
    return 0;
}
*/
/*
int main(){     ///布什, 师兄你给的这个示范界面怎么证明我用的是泛型队列啊
    printf("1.入队\n2.出队\n3.判断是否为空\n4.取队首\n5.清空队\n6.销毁队\n7.检测队长\n8.重新初始化\n9.打印队\n10.退出\n");

    Queue* queue = init_queue();
    while(1){
        int choice = 1;
        scanf("%d", choice);
        getchar();
        if(choice < 1 || choice > 10){
            printf("非法输入\n");
            continue;
        }else{
            switch (choice)
            {
            case 1:
                if(queue){
                    printf("请输入入队元素 : ");
                    char* a;
                    scanf("%s", &a);
                    getchar();
                    join_queue(queue, &a, sizeof(a));
                    printf("join succeed\n");
                }else{
                    printf("队列不存在\n");
                }
                break;
            case 2:
                if(queue){
                    char* a = *(char**)quit_queue(queue);
                printf("quit one = %s\n", a);
                }else{
                    printf("队列不存在\n");
                }
                break;
            case 3:
                if(queue){
                    if(get_queue_head(queue)){
                    printf("not empty\n");
                }
                }else{
                    printf("队列不存在\n");
                }
                break;
            case 4:
                if(queue){
                    char* a;
                a = *(char**)get_queue_head(queue);
                if(a)printf("head = %s\n", a);
                }else{
                    printf("队列不存在\n");
                }
                break;
            case 5:
                if(queue){
                    clean_queue(queue);
                }else{
                    printf("队列不存在\n");
                }
                break;
            case 6:
                if(queue){
                   free_queue(queue); 
                }else{
                    printf("队列不存在\n");
                }
                break;
            case 7:
                if(queue)printf("%d", queue->length);
                else printf("队列不存在\n");
                break;
            case 8:
                if(queue){
                    free_queue(queue);
                }
                Queue* queue = init_queue();
                printf("init succeed\n");
                break;
            case 9:
                if(queue){
                    Queuenode* p = (Queuenode*)malloc(sizeof(Queuenode));
                    p = queue->tail;
                    while(p){
                        printf("%s  ", *(char**)p->data);
                        p = p->next;
                    }
                    free(p);
                }else{
                    printf("队列不存在\n");
                }
            case 10:
                free_queue(queue);
                return 0;
            default:
                printf("illeagal input\n");
                break;
            }
        }
    _sleep(500);
    }
}
*/

int main() {
    //师兄你要求的这个菜单比本体代码还长【皱眉】【捂脸】
    printf("1.入队\n2.出队\n3.判断是否为空\n4.取队首\n5.清空队\n6.销毁队\n7.检测队长\n8.重新初始化\n9.打印队\n10.退出\n");

    Queue* queue = init_queue();
    while (1) {
        int choice;
        printf("\n请输入选项: ");
        scanf("%d", &choice);
        getchar();

        if (choice < 1 || choice > 10) {
            printf("非法输入\n");
            continue;
        }

        switch (choice) {
            case 1: {
                if (!queue) {
                    printf("队列不存在\n");
                    break;
                }
                
                // 清空输入缓冲区
                int c;
                while ((c = getchar()) != '\n' && c != EOF);
                
                printf("输入数据类型 (1.int 2.double 3.string): //想不出来其他的怎么测试");
                int type;
                if (scanf("%d", &type) != 1) {
                    printf("输入无效\n");
                    while ((c = getchar()) != '\n' && c != EOF); // 清空错误输入
                    break;
                }
                
                // 清空输入缓冲区
                while ((c = getchar()) != '\n' && c != EOF);
            
                if (type == 1) {
                    int val;
                    printf("输入整数: ");
                    if (scanf("%d", &val) == 1) {
                        join_queue(queue, &val, sizeof(int));
                        printf("整数 %d 入队成功\n", val);
                    } else {
                        printf("输入无效\n");
                    }
                } 
                else if (type == 2) {
                    double val;
                    printf("输入浮点数: ");
                    if (scanf("%lf", &val) == 1) {
                        join_queue(queue, &val, sizeof(double));
                        printf("浮点数 %f 入队成功\n", val);
                    } else {
                        printf("输入无效\n");
                    }
                } 
                else if (type == 3) {
                    char val[100];
                    printf("输入字符串: ");
                    if (scanf("%99s", val) == 1) {
                        char* str = strdup(val);
                        join_queue(queue, &str, sizeof(char*));
                        printf("字符串 \"%s\" 入队成功\n", val);
                    } else {
                        printf("输入无效\n");
                    }
                } 
                else {
                    printf("无效类型选择\n");
                }
                
                // 清空输入缓冲区
                while ((c = getchar()) != '\n' && c != EOF);
                break;
            }
            case 2: {
                if (!queue) {
                    printf("队列不存在\n");
                    break;
                }
                void* data = quit_queue(queue);
                if (data) {
                    // 实际应用中应根据类型处理，这里简单打印
                    printf("出队元素地址: %p\n", data);
                    free(data); // 释放数据内存
                }
                break;
            }
            case 3: {
                if (!queue) {
                    printf("队列不存在\n");
                    break;
                }
                printf("队列%s空\n", get_queue_head(queue) ? "非" : "为");
                break;
            }
            case 4: {
                if (!queue) {
                    printf("队列不存在\n");
                    break;
                }
                void* data = get_queue_head(queue);
                printf("队首元素地址: %p\n", data);
                break;
            }
            case 5: {
                if (queue) clean_queue(queue);
                else printf("队列不存在\n");
                break;
            }
            case 6: {
                if (queue) free_queue(queue);
                else printf("队列不存在\n");
                queue = NULL; // 避免悬空指针
                break;
            }
            case 7: {
                if (queue) printf("队长: %d\n", queue->length);
                else printf("队列不存在\n");
                break;
            }
            case 8: {
                if (queue) free_queue(queue);
                queue = init_queue(); // 重新初始化
                printf("初始化成功\n");
                break;
            }
            case 9: {
                if (!queue) {
                    printf("队列不存在\n");
                    break;
                }
                Queuenode* p = queue->tail;
                while (p) {
                    printf("元素地址: %p\n", p->data);
                    p = p->next;
                }
                break;
            }
            case 10: {
                if (queue) free_queue(queue);
                return 0;
            }
            default: {
                printf("非法选项\n");
                break;
            }
        }
    }
}