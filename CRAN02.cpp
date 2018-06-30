#include <bits/stdc++.h>

using namespace std;

int sum[20000010];

int main(){

	int t, n;
	scanf("%d",&t);
	int s = 0;
	while(t--){
		scanf("%d",&n);
		s = 0;
		memset(sum, 0, sizeof(sum));
		for (int i = 0; i < n; ++i)
		{
			int num;
			scanf("%d",&num);
			s += num;
			sum[s] += 1;
		}
	}
	return 0;
}