``` mermaid
graph LR
  A[API server] --> E{etcd};
  S[scheduler] --> A;
  K[kubelet] --> A;
   C[controller manager] --> A;
```