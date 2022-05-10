# Heterogeneous jobs:

Dans le cas du FOF, il faut que je récupère tout le monde sur un rank mpi. Problème lorsque je fais ca je fais sauter la mémoire du noeud car il y a d'autre rank qui tourne en simultané sur le noeud. On a déjà enlevé le fait qu'il fallait construire l'array complet sur chacun des ranks (cf reduce_memory_fof_catalog). Maintenant on veut tirer avantage de cela en reservant un noeud avec le rank 0 (celui qui traite toutes les données) et utiliser le maximum de processeurs sur les autres noeuds.

Ce que l'on fait pour le moment, c'est que l'on réduit le nombre de processeur au total pour que lorsque la subdivision se fait automatiquement il y est moins de process sur le noeud 0 (ou il y a le rank 0)

**IDEE** On veut réserver le noeud 0 au rank 0. (il aura toute la mémoire qu'il veut pour construire ces tableaux !)

## Heterogeneous jobs in SLURM:

On va simplement créer deux composantes lorsque l'on va lancer le job (l'une avec le rank 0 qui aura plein de mémoire pour lui) et l'autre avec les noeuds dans lesquels on pourra effectivement paralléliser le calcul au maximum (la taille mémoire utilisée ici par chaque rank est proportionnelle (dans le pire cas) au nombre de particle et donc aux nombres de rank utilisé)

TOUT est ici : https://slurm.schedmd.com/heterogeneous_jobs.html#job_steps

En particulier la section `Launch application`. Il est en particulier écrit:

    *The srun command is used to launch applications. By default, the application is launched only on the first component of a heterogeneous job, but options are available to support different behaviors.*

    *All components of a job step will have the same step ID value. If job steps are launched on subsets of the job components there may be gaps in the step ID values for individual job components.*

--> Il faut donc specifier à la commande srun qu'il faut qu'elle execute le job demander sur chacune des composantes avec `--het-group=0, 1` (si deux composantes), sinon srun n'execute la commande que sur la composante principale.

On a un gros probleme ... on ne sait pas communiquer entre les deux composantes ... --> On est peut etre sauvé avec `2.5 Dynamic Process Management` d'ici: https://manuals.plus/m/21e8c0d487022f5c029631e899de31b5e05b2e363f5e523919c305e2ee80062f#iframe -> permet de faire plein de truc le mpi2 --> en particulier de faire pop des trucs en direct avec spawn ect ...

**ATTENTION:** Ca ne marche pas !! (je ne sais pas completement pourquoi ..) Je ne peux pas communiquer entre deux composantes à cause de la version de MPI .. c'est vraiment très dérangeant ..

**REMARQUE:** SI CA MARCHE !! il ne faut pas faire `srun -het-job O,1 python test.py` mais il faut faire `srun python test.py : python test.py` (pouruqoi je ne sais pas mais ok)

**SOLUTION (NON ...):** Pas optimal (mais ca permet de passer outre sans réécrire du code pas propre) --> on va mapper directement les process pour chaque rank --> --cpu-bind=map-cu https://docs.lumi-supercomputer.eu/computing/jobs/distribution-binding/ + https://docs.nersc.gov/jobs/affinity/ pour l'architecture de CORI --> JE NE SAIS PAS SI JE PEUX FAIRE UNE COMBINAISON différente pourchaque node ...

**IDEA QUI MARCHE ?:** https://www.intel.com/content/www/us/en/developer/articles/technical/controlling-process-placement-with-the-intel-mpi-library.html et http://manpages.ubuntu.com/manpages/bionic/man1/srun.1.html--> Il y a l'option dans srun --distribution:arbitrary pour utiliser **C'EST CE QUI MARCHE** (Edmond 1 -- NERSC staff 0)


**INFO:** https://scitas-data.epfl.ch/confluence/display/DOC/CPU+affinity

**SINON** on l'écrit a la manita --> on dit que les ranks 1 --> 64 ne font rien et puis hop (mais il y aura un probleme de d'ouverture des fichiers ?)

## Memory monitoring:

Pour vérifier la mémoire utilisée dans un job, il existe des outils, surtout lorsque le job a crash ... (cf command `seff job_id`)

pour la mémoire lire (pas mal d'info qu'il n'y a pas sur la doc de NERSC) ici : https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/resource-usage/

## SLURM default parameters:

Pour avoir les paramètres par default slurm faire : `scontrol show config`

## Remarks:
    * ntask : https://stackoverflow.com/questions/39186698/what-does-the-ntasks-or-n-tasks-does-in-slurm --> ntask=2 permet de faire tourner deux commandes en simultanée sur le noeud --> si on ne veut qu'avoir qu'un rank mpi sur un noeud il faut demander une seule task
