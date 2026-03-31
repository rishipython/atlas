`benchmark`
- defines a benchmark
- gives problem to agent and evaluates it
- provides api to agent to be able to get metric for a solution (needed for ttt methods)
- needs to be generalizable (will have different benchmarks)

`agent`
- an agent that can run on the benchmark
- should take a api key for a llm (can be an openai api model or a locally hosted vllm model)
- takes problem from benchmark and constructs solution
- needs to be generalizable (will have different kinds of agents, including a base llm, different ttt agents, etc)

`setup`
- auto handle all needed setup (vllm servers, download benchmark files, etc)

`experiment`
- run experiment on benchmark
- can be used to run sweeps, etc