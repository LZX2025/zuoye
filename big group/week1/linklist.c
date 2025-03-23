#ifndef _DEFIN
#define _DEFIN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#endif

// 定义单向链表节点结构体
typedef struct SNode {
    char item;          
    struct SNode *next;     
} SNode;

// 定义双向链表节点结构体
typedef struct DNode {
    char item;          
    struct DNode *front;    
    struct DNode *next;     
} DNode;

// 定义链表结构体
typedef struct Linklist {
    void *head;             // 链表头节点
    void *tail;             // 链表尾节点
    char nodetype;          // 链表类型：'S' 表示单向链表，'D' 表示双向链表
} Linklist;

// 创建单向链表节点
SNode *create_snode(const char item) {
    SNode *node = (SNode *)malloc(sizeof(SNode));
    if (node) {
        node ->item = item;
        node->next = NULL;
    }
    return node;
}

// 创建双向链表节点
DNode *create_dnode(const char item) {
    DNode *node = (DNode *)malloc(sizeof(DNode));
    if (node) {
        node -> item = item;
        node->front = NULL;
        node->next = NULL;
    }
    return node;
}

// 初始化链表
void init_linklist(Linklist *list, char nodetype) {
    list->head = NULL;
    list->tail = NULL;
    list->nodetype = nodetype;
}

// 在链表头部插入节点
void creat_head(Linklist *list, const char item) {
    if (list->nodetype == 'S') {
        SNode *node = create_snode(item);
        if (!list->head) {
            list->head = node;
            list->tail = node;
        } else {
            node->next = (SNode *)list->head;
            list->head = node;
        }
    } else if (list->nodetype == 'D') {
        DNode *node = create_dnode(item);
        if (!list->head) {
            list->head = node;
            list->tail = node;
        } else {
            ((DNode *)list->head)->front = node;
            node->next = (DNode *)list->head;
            list->head = node;
        }
    } else {
        printf("error nodetype\n");
    }
}

// 在链表尾部插入节点
void creat_tail(Linklist *list, const char item) {
    if (list->nodetype == 'S') {
        SNode *node = create_snode(item);
        if (!list->head) {
            list->head = node;
            list->tail = node;
        } else {
            ((SNode *)list->tail)->next = node;
            list->tail = node;
        }
    } else if (list->nodetype == 'D') {
        DNode *node = create_dnode(item);
        if (!list->head) {
            list->head = node;
            list->tail = node;
        } else {
            ((DNode *)list->tail)->next = node;
            node->front = (DNode *)list->tail;
            list->tail = node;
        }
    } else {
        printf("error nodetype\n");
    }
}

// 头插法插入多个节点
void head_insert(Linklist *list, const char items[]) {
    if (list->nodetype == 'S' || list->nodetype == 'D') {
        for (int i = 0;items[i]; i++) {
            creat_head(list, items[i]);
        }
        printf("insert succeed\n");
    } else {
        printf("error nodetype\n");
    }
}

// 尾插法插入多个节点
void tail_insert(Linklist *list, const char items[]) {
    if (list->nodetype == 'S' || list->nodetype == 'D') {
        for (int i = 0; items[i]; i++) {
            creat_tail(list, items[i]);
        }
        printf("insert succeed\n");
    } else {
        printf("error nodetype\n");
    }
}
// 检查成环
int check_ring(const Linklist *list){
    int ring = 0;
    if(list->nodetype == 'S'){
        SNode *node = (SNode *)list->head;
        SNode *node_f = node;
        while(node_f){
            if(node_f->next){
                node = node ->next;
                node_f = node_f->next->next;
            }
            if(node == node_f){
                ring = 1;
                break;
            }
        }
    }else if(list->nodetype == 'D'){
        DNode *node = (DNode *)list->head;
        DNode *node_f = node;
        while(node_f){
                if(node_f->next){
                    node = node ->next;
                    node_f = node_f->next->next;
                }
                if(node == node_f){
                    ring = 1;
                    break;
                }
            }
    }else{
        printf("error type");
        ring = -1;
    }
    return ring ;
}
// 中间插入
void mid_insert(Linklist *list, const char item, const int count){
    int c = 0;
    if(list->nodetype == 'S'){
        if(check_ring(list)){
            SNode *node = (SNode*)list->head;
            SNode *node_f = node;
            while(node_f){
                
            }
        }else{

        }
    }else if(list->nodetype == 'D'){
        if(check_ring(list)){
            
        }else{

        }
    }
}
// 打印链表
void print_link(const Linklist *list) {
    if(!check_ring(list)){
        if (list->nodetype == 'S') {
            SNode *node = (SNode *)list->head;
            while (node) {
                printf("%c ", node->item);
                node = node->next;
            }
        } else if (list->nodetype == 'D') {
            DNode *node = (DNode *)list->head;
            while (node) {
                printf("%c ", node->item);
                node = node->next;
            }
        }
    }else{
        printf("the link is not fit to print");
    }
    printf("\n");
}

// 释放链表内存
void del_list(Linklist *list) {
    if (list->nodetype == 'S') {
        SNode *node = (SNode *)list->head;
        while (node) {
            SNode *temp = node;
            node = node->next;
            free(temp);
        }
    } else if (list->nodetype == 'D') {
        DNode *node = (DNode *)list->head;
        while (node) {
            DNode *temp = node;
            node = node->next;
            free(temp);
        }
    }
    list->head = NULL;
    list->tail = NULL;
}
// 建立循环
int creat_ring(Linklist *list, int num){
    int suc = 0;
    int n = 0;
    if(!check_ring(list)){
        if (list->nodetype == 'S') {
            SNode *node = (SNode *)list->head;
            while(node){
                if(n == num){
                    list->tail = node;
                    suc = 1;
                    break;
                }else{
                    node = node->next;
                    n++;
                }
            }
        } else if (list->nodetype == 'D') {
            DNode *node = (DNode *)list->head;
            while(node){
                if(n == num){
                    list->tail = node;
                    suc = 1;
                    break;
                }else{
                    node = node->next;
                    n++;
                }
            }
        }
    }else{
        printf("the linklist has ring");
    }
    return suc;
}
/*
示例：
int main() {
    Linklist list;
    init_linklist(&list, 'S'); // 初始化单向链表

    const char items[] = {"ABC"};
    head_insert(&list, items); // 头插法插入节点
    print_link(&list); // 打印链表

    tail_insert(&list, items); // 尾插法插入节点
    print_link(&list); // 打印链表

    del_list(&list); // 释放链表内存
    return 0;
}
    说明书：

        链表储存char类型数据（不觉得后面要用就没做泛型）

        用于创建节点
        SNode *create_snode(const char item);单向
        DNode *create_dnode(const char item);双向
        返回头指针

        void init_linklist(Linklist *list, char nodetype);初始化， 传入"S"(单向) or "D"（双向）

        添加节点
        void creat_head(Linklist *list, const char item);
        void creat_tail(Linklist *list, const char item);加一个
        void head_insert(Linklist *list, const char items[]);
        void tail_insert(Linklist *list, const char items[]);加一串

        void print_link(const Linklist *list);字面意思from  head  ——>  tail
        void del_list(Linklist *list);

        int check_ring(const Linklist *list);检查是否成环， 返回0（否）1（是）
        int creat_ring(Linklist *list, int num);将尾指向第num个节点处（从0计数）
*/