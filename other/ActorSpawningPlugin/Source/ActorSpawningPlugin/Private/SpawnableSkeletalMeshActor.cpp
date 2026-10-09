#include "SpawnableSkeletalMeshActor.h"

#include "UObject/Object.h"
#include "Engine/EngineTypes.h"


DEFINE_LOG_CATEGORY(SpawnableSkeletalMeshActorLog);

ASpawnableSkeletalMeshActor::ASpawnableSkeletalMeshActor()
{
	UE_LOG(SpawnableSkeletalMeshActorLog, Log, TEXT("CTOR"));

	// Set this actor to call Tick() every frame. 
	// You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

    BaseMeshComponent = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("BaseMesh"));
	RootComponent = BaseMeshComponent;
    //BaseMeshComponent->SetupAttachment(RootComponent);
}

ASpawnableSkeletalMeshActor::~ASpawnableSkeletalMeshActor() {
}

void ASpawnableSkeletalMeshActor::SetSpawnableSkeletalMesh(USkeletalMesh* skeletalMesh) 
{
	UE_LOG(SpawnableSkeletalMeshActorLog, Log, TEXT("SetSpawnableSkeletalMesh()"));

	BaseMesh = skeletalMesh;

    BaseMeshComponent->SetSkeletalMesh(BaseMesh);

	BaseMeshComponent->SetCollisionObjectType(ECollisionChannel::ECC_PhysicsBody);
	BaseMeshComponent->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	BaseMeshComponent->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);

	BaseMeshComponent->SetSimulatePhysics(true);
}

