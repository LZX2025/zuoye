class SNode :
    """单项链表"""
    def __init__(self, item) :
        self.item = item
        self.next = None

class DNode :
    """双向链表"""
    def __init__(self, item) :
        self.item = item
        self.front = None
        self.next = None


class Linklist :
    def __init__(self, nodetype):
        """S单向 D双向"""
        self.head = None
        self.tail = None
        self.nodetype = nodetype #链表类型

    def creat_head(self, item) :
        """加头"""
        if self.nodetype == 'S' :
            node = SNode(item)
            if not self.head:
                self.head = node
                self.tail = self.head
            else:
                node.next = self.head
                self.head = node
            return self.head
        elif self.nodetype == 'D' :
            node = DNode(item)
            if not self.head:
                self.head = node
                self.tail = self.head
            else:
                node.next = self.head
                self.head.front = node
                self.head = node
            return self.head
        #elif
        else :
            print("error nodetype")


    def creat_tail(self, item) :
        """加尾"""
        if self.nodetype == 'S' :
            node = SNode(item)
            if not self.head:
                self.head = node
                self.tail = self.head
            else:
                self.tail.next = node
                self.tail = node
            return self.head
        elif self.nodetype == 'D' :
            node = DNode(item)
            if not self.head:
                self.head = node
                self.tail = self.head
            else:
                self.tail.next = node
                node.front = self.tail
                self.tail = node
            return self.head
        #elif
        else :
            print("error nodetype")
    

    def head_insert(self, ilist) :
        """头插法, itemlist列表"""
        if self.nodetype == 'S' :
            if not self.head : 
                self.head = SNode(ilist[0])
                for e in ilist[1:] :
                    self.head = self.creat_head(e, self.nodetype)
            else :
                for e in ilist :
                    self.head = self.creat_head(e, self.nodetype)
            print("insert succeed")
        elif self.nodetype == 'D' :
            if not self.head :
                self.head = DNode(ilist[0])
                for e in ilist[1:] :
                    self.head = self.creat_head(e, self.nodetype)
            else :
                for e in ilist :
                    self.head = self.creat_head(e, self.nodetype)
            print("insert succeed")
        else :
            print("error Nodetype")

        

    def tail_insert(self, ilist) :
        """尾插法, S单向, D双向, itemlist列表"""
        if self.nodetype == 'S' :
            if not self.head :
                self.head = SNode(ilist[0])
                self.tail = self.head
                for e in ilist[1:] :
                    self.creat_tail(e)
            else :
                for e in ilist :
                    self.creat_tail(e)
            print("insert succeed")
        elif self.nodetype == 'D' :
            if not self.head :
                self.head = DNode(ilist[0])
                self.tail = self.head
                for e in ilist[1:] :
                    self.creat_tail(e)
            else :
                for e in ilist :
                    self.head = self.creat_tail(e)
            print("insert succeed")
        else :
            print("error Nodetype")


    def mid_insert(self, ilist, pos) :
        """插入， ilist(list)  , pos(node)"""
        if self.nodetype == 'S' :
            if not self.head :
                self.head = SNode(ilist[0])
                self.tail = self.head
                for e in ilist[1:] :
                    self.creat_tail(e)
                pos = self.head
                print("no list_head, init pos = head")
                print("insert succeed")
            else :
                node = self.head
                fin = 0
                if self.check_ring() :  # 检查是否成环
                    node_x = node
                    while node :
                        node = node.next
                        node_x = node_x.next.next
                        if node == pos :
                            fin = 1
                            break
                        elif node == node_x :
                            print("unfined pos")
                            node = None
                            break
                else :
                    while node :
                        node = node.next
                        if node == pos :
                            fin = 1
                            break
                    if not fin :
                        print("unfined pos")
                        node = None
                            
                if node :
                    for e in ilist :
                        node = node.next
                        node.item = SNode(e)
                    print("insert succeed")
                    pass

        elif self.nodetype == 'D' :
            if not self.head :
                self.head = DNode(ilist[0])
                self.tail = self.head
                for e in ilist[1:] :
                    self.creat_tail(e)
                pos = self.head
                print("no list_head, init pos = head")
                print("insert succeed")
            else :
                node = self.head
                fin = 0
                if self.check_ring() :  # 检查是否成环
                    node_x = node
                    while node :
                        node = node.next
                        node_x = node_x.next.next
                        if node == pos :
                            fin = 1
                            break
                        elif node == node_x :
                            print("unfined pos")
                            node = None
                            break
                else :
                    while node :
                        node = node.next
                        if node == pos :
                            fin = 1
                            break
                    if not fin :
                        print("unfined pos")
                        node = None
                            
                if node :
                    for e in ilist :
                        p = node
                        node = node.next
                        node.item = DNode(e)
                        node.front = p
                    print("insert succeed")
                    pass

        else :
            print("error nodetype")


    def load_link(self) :
        """load as list"""
        node = self.head 
        link_list = []
        if self.check_ring() :
            print("The linklist has ring and not fit to load")
        else :           
            while node :
                link_list.append(node.item)
                node = node.next
        return link_list
    

    def print_link(self) :
        """直接print"""
        node = self.head
        if self.check_ring() :
            print("The linklist has ring and not fit to print")
        else :           
            while node :
                print(node.item,end=' ')
                node = node.next
            print()


    def check_ring(self) :
        """True--has_ring ; False--none_ring"""
        node = self.head
        node_f = self.head
        ring = 0
        if node.next and node :
            while node_f :
                node = node.next
                if node_f.next :
                    node_f = node_f.next.next
                elif node_f :
                    node_f = node_f.next
                else :
                    node_f = None
                if node_f == node :
                    ring = 1
                    break
        return ring


    def search_item(self, item) :
        """return target_front_node"""
        node = self.head
        fin = 0
        if self.check_ring() :
            node_x = node.next
            while node and node != node_x :
                if node.next.item == item :
                    fin = 1
                    return node
                else :
                    node = node.next
                    node_x = node_x.next.next
        else :
            while node :
                if node.next.item == item :
                    fin = 1
                    return node
                else :
                    node = node.next
        if not fin and self.head.item == item :
            print("link has one node")
            return self.head
        elif not fin :
            print("unfined item")
            return None


    def del_node(self, pos) :
        """del pos(node)"""
        node = self.head
        fin = 0
        if self.nodetype == 'D' :
            if pos and pos.next.next :
                pos.next = pos.next.next
                pos.next.front = pos
            del pos
            fin = 1
            print("delete succeed")
        elif self.nodetype == 'S' :
            if pos and pos.next.next :
                pos.next = pos.next.next
                del pos
                fin = 1
                print("delete succeed")
        if not fin :
            print("unfined pos")
 

    def del_list(self):
        """del all linklist"""
        node = self.head
        while node.next :
            p = node
            node = node.next
            del p
        del node

    def creat_ring(self, pos) :
        """link tail and pos"""
        if self.nodetype == 'S' or self.nodetype == 'D' :
            self.tail.next = pos
        #elif
        else :
            print("wrong nodetype")




            
#TEST________________

link1 = Linklist('D')
link1.tail_insert([1,2,3,4,5,6])
list1 = link1.load_link()
print(list1)
print(link1.head.item)
link1.print_link()
link1.del_node(link1.search_item(3))
link1.print_link()
link1.creat_ring(link1.search_item(2))
#print(link1.check_ring)
link1.print_link()





    

