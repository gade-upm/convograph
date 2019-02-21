# -*- coding: utf-8 -*-

from scipy.sparse import coo_matrix
from collections import defaultdict

import graph_tool.all as gt
import numpy as np
import itertools
import time
import pdb

def generate_toy():
    g = gt.Graph()
    g.ep['peso_arco'] = g.new_edge_property('float')
    
    v1 = g.add_vertex()
    v2 = g.add_vertex()
    v3 = g.add_vertex()
    
    g.add_edge_list([(v1,v2), (v1,v3)])
    
    for e in g.edges():
        g.ep['peso_arco'][e] = 2
    
    return g

def generate_graph(arcos_fuertes,arcos_debiles):
    
    '''
    A los arcos fuertes se les asigna un peso 100 veces mayor que a los arcos débiles
    '''
    g=gt.Graph(directed=True)
    # etiqueta_nodo=g.new_vertex_property('string')
    g.vp['etiqueta_nodo']=g.new_vertex_property('string')

    num_vertices = len(set(zip(*arcos_fuertes)[0]).union(set(zip(*arcos_fuertes)[1]
            ).union(set(zip(*arcos_debiles)[0]).union(set(zip(*arcos_debiles)[1])))))

    for i in range(num_vertices):
        u=g.add_vertex()
        g.vp['etiqueta_nodo'][u]=str(i)


    # etiqueta_arco=g.new_edge_property('string')
    g.ep["etiqueta_arco"]=g.new_edge_property('string')

    # peso_arco=g.new_edge_property('float')
    g.ep["peso_arco"]=g.new_edge_property('float')

    for j in arcos_fuertes:
        u=gt.find_vertex(g,g.vp['etiqueta_nodo'],str(j[0]))
        v=gt.find_vertex(g,g.vp['etiqueta_nodo'],str(j[1]))
        e=g.add_edge(u[0],v[0])

        a=100*np.random.rand()
        if(a<10):
            a=10+(10*np.random.rand())
        g.ep['etiqueta_arco'][e]=str(round(a,1))
        g.ep['peso_arco'][e]=round(a,1)

    for j in arcos_debiles:
        u=gt.find_vertex(g,g.vp['etiqueta_nodo'],str(j[0]))
        v=gt.find_vertex(g,g.vp['etiqueta_nodo'],str(j[1]))
        e=g.add_edge(u[0],v[0])

        a=np.random.rand()
        g.ep['etiqueta_arco'][e]=str(round(a,1))
        g.ep['peso_arco'][e]=round(a,1)

    # gt.graph_draw(g,vertex_text=etiqueta_nodo,edge_text=etiqueta_arco,vertex_size=8)
    return g


def generate_graph_2():
    arcos_fuertes=[(1,2),(2,3),(3,4),(2,4),(4,5),(1,4),(5,6),(6,3),(1,5),(10,15),(15,11),(12,13),(11,12),(11,10),(12,15),(11,13)]
    arcos_debiles=[(2,7),(3,8),(4,9),(7,10),(8,11),(9,12),(1,16),(6,17),(5,18),(14,15),(13,19),(0,2)]
    return generate_graph(arcos_fuertes,arcos_debiles)

def generate_graph_3():
    arcos_fuertes=[(0,1),(1,2),(2,3),(0,3),(4,5),(5,6),(8,10)]
    arcos_debiles=[(0,6),(4,6),(0,7),(4,7),(5,8),(9,2),(3,1),
                   (10,5),(11,2),(12,3),(12,11),(7,3),(1,6),(1,3),
                   (0,4),(12,7),(12,0),(11,4),(6,2)]
    return generate_graph(arcos_fuertes,arcos_debiles)

def grafo_dual(g):

    '''
    El arco de entrada (g) debe tener los atributos de arco "peso_arco" y "etiqueta_arco"
    y de nodo "etiqueta_nodo" con los siguientes tipos:


    etiqueta_nodo=g.new_vertex_property('string')
 g.vertex_properties["etiqueta_nodo"]=etiqueta_nodo

 etiqueta_arco=g.new_edge_property('string')
 g.edge_properties["etiqueta_arco"]=etiqueta_arco

 peso_arco=g.new_edge_property('float')
 g.edge_properties["peso_arco"]=peso_arco

    El grafo dual (h) que se devuelve tiene los atributos "etiqueta_nodo", "etiqueta_arco", "pares" y "peso"
    que se detallan en el código.
    Algunos de estos atributos son sólo a efectos de visualización

    El grafo dual es el grafo NO DIRIGIDO obtenido al pasar los arcos a nodos
    y los nuevos arcos vendrán de nodos intersección entre dos arcos originales
    '''

    h=gt.Graph(directed=False)

    etiqueta_nodo=h.new_vertex_property('string')   #Nombre del nodo en el grafo dual y la cantidad económica del arco primal que representa
    h.vertex_properties["etiqueta_nodo"]=etiqueta_nodo

    etiqueta_arco=h.new_edge_property('string')  # peso del arco dual (en el grafo dual)=diferencia entre nodos duales
    h.edge_properties["etiqueta_arco"]=etiqueta_arco

    pares=h.new_vertex_property('vector<float>') #Guardo aquí los pares de nodos primales que que representa cada nodo dual.
    h.vertex_properties["pares"]=pares

    peso=h.new_vertex_property('float')  # Valor económico del arco primal en el nodo dual que lo representa.
    h.vertex_properties["peso"]=peso


    k=0
    for e in g.edges():
        v=h.add_vertex()
        h.vp.etiqueta_nodo[v]='e'+str(k)+'-'+str(g.ep.etiqueta_arco[e])
        h.vp.pares[v]=(float(g.vp.etiqueta_nodo[e.source()]),float(g.vp.etiqueta_nodo[e.target()]))
        h.vp.peso[v]=float(g.ep.peso_arco[e])
        k=k+1

    arcos_visitados=[]
    # visitor = DFSBetaSearch()
    # gt.dfs_search(g, None,visitor)
    for v in h.vertices():
        for w in h.vertices():
            if (v,w) not in arcos_visitados and (w,v) not in arcos_visitados and v!=w and len(set([h.vp.pares[w][0],h.vp.pares[w][1]]).intersection(set([h.vp.pares[v][0],h.vp.pares[v][1]])))>0:
                f=h.add_edge(v,w)
                diferencia_nodos=np.abs(float(h.vp.peso[v])-float(h.vp.peso[w]))
                h.ep.etiqueta_arco[f]=str(diferencia_nodos)
                arcos_visitados.append((v,w))


#    for v in h.vertices():
#        print h.vp.etiqueta_nodo[v], h.vp.pares[v]

    return h

def load_graph(type='graph2'):
    graphs = {
      'toy': generate_toy(),
      'graph2': generate_graph_2(),
      'graph3': generate_graph_3(),
      'cond-mat-2003': gt.collection.data['cond-mat-2003']
    }
    
    return graphs[type]

def compute_combs(lst):
    groups = defaultdict(set)
    for tpl in lst:
        for v in tpl:
            groups[v].add(tpl)
    r = [list(tuplas) for n, tuplas in groups.items() if len(tuplas)> 1]
    
    return r

def create_graph(g, node_label, edge_label):
    h = gt.Graph(directed=False)
    h.ep["peso_arco"] = h.new_edge_property('float')
    
    edges = [(g.vertex_index[e.source()], g.vertex_index[e.target()]) for e in g.edges()]
    
    combs = compute_combs(edges)
    
    vertices = h.add_vertex(n=len(edges))
    mapper = dict()
    mapper_inv = dict()
    for v in vertices:
        mapper[edges[g.vertex_index[v]]] = g.vertex_index[v]
        mapper_inv[g.vertex_index[v]] = edges[g.vertex_index[v]]
    
    links = set([(mapper[s],mapper[t]) for comp in combs for s,t in itertools.combinations(comp, 2)])
    
    h.add_edge_list(links)
    
    # TODO: Mejorar insercion de valores
    # Prop.a = numpy.array
    for e in h.edges():
        e1_i, e1_f = mapper_inv[h.vertex_index[e.source()]]
        e2_i, e2_f = mapper_inv[h.vertex_index[e.target()]]
        h.ep['peso_arco'][e] = np.abs(float(float(g.ep[edge_label][g.edge(e1_i, e1_f)]))-float(float(g.ep[edge_label][g.edge(e2_i, e2_f)])))
    
    return h
    
   

def build_inducido(g, node_label, edge_label):
    h = create_graph(g, node_label, edge_label)

    return h
    
      
def main():
    g = load_graph(type='graph3')
    print(g)
#    s0 = time.time()
#    h = grafo_dual(g)
#    s = time.time()
#    print(h)
#    print('{0}s'.format(s-s0))
    s0 = time.time()
    h = build_inducido(g, "etiqueta_nodo", "peso_arco")
    s = time.time()
    print(h)
    print('{0}s'.format(s-s0))

if __name__ == '__main__':
    main()
