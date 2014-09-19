#ifndef INTERFACES_CPU_H_
#define INTERFACES_CPU_H_
#endif /* INTERFACES_CPU_H_ */

struct Performances Random_sched_try(int node_number, int f, float drop_prob, void *genV, int write_descriptor);
void Init_simulation_parameters();
struct Performances Static_grid(int node_number, int f, float drop_prob, void *gen, int write_descriptor);
