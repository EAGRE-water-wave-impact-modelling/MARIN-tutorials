## Benney-Luke equations Firedrake FEM

This folder contains the following material:
- `bennylukefbVP.py`: the associated [*Firedrake*](https://www.firedrakeproject.org/) code for solving the worked example. nVP=0 standard weak formulation as on Firedrake site using Stormer-Verlet in time; nVP=1 VP/minimisation problem modified mid-point (MMP) in time; and, nVP=2 for nonlinear SWE with MMP in time.
- `energyplotBL.py`: code for plotting the energy over time.
- :new: `benneylukefbwbVP.py`: wave breaking parameterisation for SWE; nVP=2, in progress but the mixed minimization/VP and viscous combination is novel and useful. See section 3.1 in updated :new: `ex23_3wavedynFDwaveb.pdf` Note that runs with flags nonno=0,3,4 work best to worse. More smoothing is required; notte the factor 4.0 in nonno=0 (e.g. set it back to 1.0 and check).

## How to run the *Firedrake* code from a terminal?
1. Navigate to the Apptainer image folder from the terminal:
   ```
   cd /localhome/data/sw/firedrake-2023/
   ```
   
2. Activate the Apptainer image:
   ```
   apptainer shell -B ~/.cache:/home/firedrake/firedrake/.cache ./firedrake_latest.sif
   ```
   Now your terminal will be running inside the firedrake_latest container. You will only have access to modules that are installed inside the          container other modules including paraview will have to be run from another terminal.

4. Activate Firedrake:
   ```
   source /home/firedrake/firedrake/bin/activate
   ```
   
5. Navigate to the directory where you put the code (say `~/Desktop/FEM_course/BLeqns/`):
    ```
    cd Desktop/FEM_course/BLeqns/
    ```
6. Execute the code:
    ```
    python3 bennylukefbVP.py.py
    ```
7. Check the results:
- energy plot in the data directory:
```
python3 energyplotBL.py
```
- in-code generated cross-sectional channel profiles at a certain y-position.
- use Paraview to display the 2D fields which can be loaded from a terminal (be careful, please, when Anaconda and/or Fluent are open, these may interfere):
     ```
     module load paraview-rhel8/5.11.2
     paraview
     ```

8. Further information: see `ex23_3wavedynFD.pdf`
   
