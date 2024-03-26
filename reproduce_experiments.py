



## Generate datasets
exec(open("generate_data.py").read())

## Reproduces experiments on exact recovery
exec(open("testscript_recovery_grad_assumption.py").read())

## Reproduce experiments on convergence rate
exec(open("testscript_rate.py").read())

## Reproduce experiments on cameraman image
exec(open("testscript_cameraman.py").read())

## Generate figures from the above-computed results
exec(open("make_figures_paper.py").read())



