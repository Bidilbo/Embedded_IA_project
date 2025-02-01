# Embedded_IA_project

Ce dépot permet de lancer l'apprentissage d'un MLP quelconque, puis d'exporter les poids dans un fichier .json. Le makefile fourni permet ensuite de réaliser l'inférence à partir des images de test. 

## Créer l'image Docker

Pour lancer le projet, créer l'image Docker "student" à partir du Dockerfile si ce n'est pas déjà fait. 

## Lancer l'entrainement

Executer la commande : 
```bash
./deploy.sh
```
## Lancer l'inférence

Se placer dans le dossier C. Executer les commandes :
```bash
make veryclean
make
./all ../images_inference/image.bmp
```

