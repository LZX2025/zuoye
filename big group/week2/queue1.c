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
        queue->head = NULL; // �����ѿ�
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

    // ���Կն���
    assert(get_queue_head(queue) == NULL);

    // ���
    join_queue(queue, &a1, sizeof(a1));
    join_queue(queue, &a2, sizeof(a2));
    join_queue(queue, a3, sizeof(a3));

    // ����
    int* b1 = (int*)quit_queue(queue);
    double* b2 = (double*)quit_queue(queue);
    char* b3 = (char*)quit_queue(queue);
    
    if (b1) printf("%d\n", *b1);
    if (b2) printf("%f\n", *b2);
    if (b3) printf("%s\n", b3);

    // �ͷ��ڴ�
    free(b1);
    free(b2);
    free(b3);

    // ������Ӳ���
    join_queue(queue, &a1, sizeof(a1));
    int* temp = (int*)get_queue_head(queue);
    if (temp) printf("temp: %d\n", *temp);

    free_queue(queue);
    return 0;
}
*/
/*
int main(){     ///��ʲ, ʦ����������ʾ��������ô֤�����õ��Ƿ��Ͷ��а�
    printf("1.���\n2.����\n3.�ж��Ƿ�Ϊ��\n4.ȡ����\n5.��ն�\n6.���ٶ�\n7.���ӳ�\n8.���³�ʼ��\n9.��ӡ��\n10.�˳�\n");

    Queue* queue = init_queue();
    while(1){
        int choice = 1;
        scanf("%d", choice);
        getchar();
        if(choice < 1 || choice > 10){
            printf("�Ƿ�����\n");
            continue;
        }else{
            switch (choice)
            {
            case 1:
                if(queue){
                    printf("���������Ԫ�� : ");
                    char* a;
                    scanf("%s", &a);
                    getchar();
                    join_queue(queue, &a, sizeof(a));
                    printf("join succeed\n");
                }else{
                    printf("���в�����\n");
                }
                break;
            case 2:
                if(queue){
                    char* a = *(char**)quit_queue(queue);
                printf("quit one = %s\n", a);
                }else{
                    printf("���в�����\n");
                }
                break;
            case 3:
                if(queue){
                    if(get_queue_head(queue)){
                    printf("not empty\n");
                }
                }else{
                    printf("���в�����\n");
                }
                break;
            case 4:
                if(queue){
                    char* a;
                a = *(char**)get_queue_head(queue);
                if(a)printf("head = %s\n", a);
                }else{
                    printf("���в�����\n");
                }
                break;
            case 5:
                if(queue){
                    clean_queue(queue);
                }else{
                    printf("���в�����\n");
                }
                break;
            case 6:
                if(queue){
                   free_queue(queue); 
                }else{
                    printf("���в�����\n");
                }
                break;
            case 7:
                if(queue)printf("%d", queue->length);
                else printf("���в�����\n");
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
                    printf("���в�����\n");
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
    //ʦ����Ҫ�������˵��ȱ�����뻹������ü����������
    printf("1.���\n2.����\n3.�ж��Ƿ�Ϊ��\n4.ȡ����\n5.��ն�\n6.���ٶ�\n7.���ӳ�\n8.���³�ʼ��\n9.��ӡ��\n10.�˳�\n");

    Queue* queue = init_queue();
    while (1) {
        int choice;
        printf("\n������ѡ��: ");
        scanf("%d", &choice);
        getchar();

        if (choice < 1 || choice > 10) {
            printf("�Ƿ�����\n");
            continue;
        }

        switch (choice) {
            case 1: {
                if (!queue) {
                    printf("���в�����\n");
                    break;
                }
                
                // ������뻺����
                int c;
                while ((c = getchar()) != '\n' && c != EOF);
                
                printf("������������ (1.int 2.double 3.string): //�벻������������ô����");
                int type;
                if (scanf("%d", &type) != 1) {
                    printf("������Ч\n");
                    while ((c = getchar()) != '\n' && c != EOF); // ��մ�������
                    break;
                }
                
                // ������뻺����
                while ((c = getchar()) != '\n' && c != EOF);
            
                if (type == 1) {
                    int val;
                    printf("��������: ");
                    if (scanf("%d", &val) == 1) {
                        join_queue(queue, &val, sizeof(int));
                        printf("���� %d ��ӳɹ�\n", val);
                    } else {
                        printf("������Ч\n");
                    }
                } 
                else if (type == 2) {
                    double val;
                    printf("���븡����: ");
                    if (scanf("%lf", &val) == 1) {
                        join_queue(queue, &val, sizeof(double));
                        printf("������ %f ��ӳɹ�\n", val);
                    } else {
                        printf("������Ч\n");
                    }
                } 
                else if (type == 3) {
                    char val[100];
                    printf("�����ַ���: ");
                    if (scanf("%99s", val) == 1) {
                        char* str = strdup(val);
                        join_queue(queue, &str, sizeof(char*));
                        printf("�ַ��� \"%s\" ��ӳɹ�\n", val);
                    } else {
                        printf("������Ч\n");
                    }
                } 
                else {
                    printf("��Ч����ѡ��\n");
                }
                
                // ������뻺����
                while ((c = getchar()) != '\n' && c != EOF);
                break;
            }
            case 2: {
                if (!queue) {
                    printf("���в�����\n");
                    break;
                }
                void* data = quit_queue(queue);
                if (data) {
                    // ʵ��Ӧ����Ӧ�������ʹ�������򵥴�ӡ
                    printf("����Ԫ�ص�ַ: %p\n", data);
                    free(data); // �ͷ������ڴ�
                }
                break;
            }
            case 3: {
                if (!queue) {
                    printf("���в�����\n");
                    break;
                }
                printf("����%s��\n", get_queue_head(queue) ? "��" : "Ϊ");
                break;
            }
            case 4: {
                if (!queue) {
                    printf("���в�����\n");
                    break;
                }
                void* data = get_queue_head(queue);
                printf("����Ԫ�ص�ַ: %p\n", data);
                break;
            }
            case 5: {
                if (queue) clean_queue(queue);
                else printf("���в�����\n");
                break;
            }
            case 6: {
                if (queue) free_queue(queue);
                else printf("���в�����\n");
                queue = NULL; // ��������ָ��
                break;
            }
            case 7: {
                if (queue) printf("�ӳ�: %d\n", queue->length);
                else printf("���в�����\n");
                break;
            }
            case 8: {
                if (queue) free_queue(queue);
                queue = init_queue(); // ���³�ʼ��
                printf("��ʼ���ɹ�\n");
                break;
            }
            case 9: {
                if (!queue) {
                    printf("���в�����\n");
                    break;
                }
                Queuenode* p = queue->tail;
                while (p) {
                    printf("Ԫ�ص�ַ: %p\n", p->data);
                    p = p->next;
                }
                break;
            }
            case 10: {
                if (queue) free_queue(queue);
                return 0;
            }
            default: {
                printf("�Ƿ�ѡ��\n");
                break;
            }
        }
    }
}