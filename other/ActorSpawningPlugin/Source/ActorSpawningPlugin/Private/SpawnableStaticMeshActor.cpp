#include "SpawnableStaticMeshActor.h"

#include "UObject/Object.h"

DEFINE_LOG_CATEGORY(SpawnableStaticMeshActorLog);

ASpawnableStaticMeshActor::ASpawnableStaticMeshActor()
{
	// Set this actor to call Tick() every frame. 
	// You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

    BaseMeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("BaseMesh"));
    BaseMeshComponent->SetupAttachment(RootComponent);
}

ASpawnableStaticMeshActor::~ASpawnableStaticMeshActor() {
}

void ASpawnableStaticMeshActor::SetSpawnableStaticMesh(UStaticMesh* staticMesh) 
{
	BaseMesh = staticMesh;
    BaseMeshComponent->SetStaticMesh(BaseMesh);
	BaseMeshComponent->SetSimulatePhysics(true);
	BaseMeshComponent->UpdateCollisionFromStaticMesh();
}

