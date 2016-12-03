#include<cstdio>
#include<cstring>
#include<iostream>
#include<queue>
#include<algorithm>
#include<stack>
#include<vector>
#include<cmath>
#include<unordered_set>
using namespace std;
const int MAXM = 10000 + 10;
const int MAXN = 1000 + 10;
const int MAXNODE = MAXM + MAXN + 2;
const int INF = 0x7fffffff;

struct Edge {
	int to;
	int cap;
	int rev; // ·´Ïò±ß
	Edge(int to, int cap, int rev) :to(to), cap(cap), rev(rev) {}
};
vector<Edge> g[MAXNODE];
vector<Edge> g2[MAXNODE];
int level[MAXNODE];

void add_edge(int from,int to,int cap) {
	g[from].push_back(Edge(to,cap,g[to].size()));
	g[to].push_back(Edge(from, 0, g[from].size() - 1));
}

void bfs(int from, int to) {
	memset(level, -1, sizeof(level));
	queue<int> q;
	q.push(from);
	level[from] = 0;
	while (!q.empty()) {
		int cur = q.front();
		q.pop();
		for (Edge &e : g[cur]) {
			if (e.cap > 0 && level[e.to] < 0) {
				q.push(e.to);
				level[e.to] = level[cur] + 1;
			}
		}
	}
}

int dfs(int from, int to,int flow) {
	if (from == to) return flow;
	for (Edge &e : g[from]) {
		if (e.cap >0 && level[from] < level[e.to]) {
			int f = dfs(e.to, to, min(flow, e.cap));
			if (f > 0) {
				e.cap -= f;
				g[e.to][e.rev].cap += f;
				return f;
			}
		}
	}
	return 0;
}


int max_flow(int from, int to) {
	int flow = 0,f=0;
	while (1) {
		bfs(from, to);
		if (level[to] < 0) return flow;
		while (f = dfs(from, to, INF)) {
			flow += f;
		}
	}
	return flow;
}

int binary_search(int s,int t,int m,int n) {
	int L = 1, R = m + 1;
	while (L < R) {
		int mid = (L+R)>> 1;
		for (int i = m; i < m + n; i++) 
			add_edge(i, t, mid);

		if (max_flow(s, t) == m) R = mid;
		else L = mid + 1;

		for (int i = 0; i < m + n + 2; i++) g[i] = g2[i];
	}
	return L;
}


int main()
{
	int T;
	int n, m, x, y;
	while (cin>>m>>n)
	{
		int s = n + m, t = n + m + 1;
		for (int i = 0; i < m + n + 2; i++) g[i].clear();
		for (int i = 0; i < m; i++) {
			scanf("%d%d", &x, &y);
			add_edge(i, x+m - 1, 1);
			add_edge(i, y+m - 1, 1);
			add_edge(s, i, 1);
		}
		for (int i = 0; i < m + n + 2; i++) g2[i] = g[i];
		printf("%d\n", binary_search(s, t, m, n));
	}
	return 0;
}