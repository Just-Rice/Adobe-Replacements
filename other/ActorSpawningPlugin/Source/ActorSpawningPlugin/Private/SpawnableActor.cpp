#include "SpawnableActor.h"

#include "UObject/Object.h"

DEFINE_LOG_CATEGORY(SpawnableActorLog);

ASpawnableActor::ASpawnableActor()
{
	UE_LOG(SpawnableActorLog, Log, TEXT("CTOR()"));

	// Set this actor to call Tick() every frame. 
	// You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;
}

ASpawnableActor::~ASpawnableActor()
{
}

void ASpawnableActor::BeginPlay()
{
	Super::BeginPlay();
}

void ASpawnableActor::BeginDestroy()
{
	Super::BeginDestroy();
}

void ASpawnableActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void ASpawnableActor::StartDestroyTimer(float seconds)
{
	GetWorldTimerManager().SetTimer(DestroyTimer, this, &ASpawnableActor::InvokeDestroy, seconds);
}

void ASpawnableActor::InvokeDestroy()
{
	Destroy();
}
