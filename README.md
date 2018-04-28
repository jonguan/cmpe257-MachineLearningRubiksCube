# cmpe257-MachineLearningRubiksCube

## Setup / Intallation Instructions
1. Install anaconda. https://docs.anaconda.com/anaconda/install/
2. Verify that anaconda is installed correctly.
`conda list` should list out all of the libraries.
3. Create the conda virtual env.  https://conda.io/docs/user-guide/tasks/manage-environments.html
Make sure you activate the environment too. https://conda.io/docs/user-guide/tasks/manage-environments.html#activating-an-environment  
4. Install VirtualEnv  
`pip install --upgrade tensorflow `
5. Install Keras  
`pip install --upgrade keras`
6.  Project may complain that python needs to run as an app.  Put this in .bashrc
```
function frameworkpython {
    if [[ ! -z "$VIRTUAL_ENV" ]]; then
        PYTHONHOME=$VIRTUAL_ENV /usr/local/bin/python "$@"
    else
        /usr/local/bin/python "$@"
    fi
}
```
7. `source .bashrc`
8. There will be other libraries that may have to be installed.  Install them not in the conda env. but in the desktop env.


## Magic Cube
Magic cube is a rubiks cube visualizer.  It will serve as the model for our project.
https://github.com/davidwhogg/MagicCube
