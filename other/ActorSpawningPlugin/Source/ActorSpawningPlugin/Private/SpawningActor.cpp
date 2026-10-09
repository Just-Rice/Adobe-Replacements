#include "SpawningActor.h"

#include "SpawnableActor.h"

#include "Engine/World.h"
#include "UObject/Object.h"
#include "Math/UnrealMathUtility.h"


DEFINE_LOG_CATEGORY(SpawningActorLog);

ASpawningActor::ASpawningActor()
{
	UE_LOG(SpawningActorLog, Log, TEXT("CTOR()"));
}

ASpawningActor::~ASpawningActor()
{
}

void ASpawningActor::BeginPlay()
{
	Super::BeginPlay();
}

void ASpawningActor::BeginDestroy()
{
	Super::BeginDestroy();
}

ASpawnableStaticMeshActor* ASpawningActor::SpawnStaticMeshActor()
{
	// Spawn actor
	// TODO(bt): This will memory leak.
	ASpawnableStaticMeshActor* spawnedActor = (ASpawnableStaticMeshActor*)GetWorld()->SpawnActor(ASpawnableStaticMeshActor::StaticClass());

	FVector playerLocation = LookupPlayerLocation();

	// Teleport actor randomly
	// TODO(bt): Put this in front of the camera, or on the (camera -> player) arc!
	float x = FMath::RandRange(-4000.0f, 4000.0f);
	float y = FMath::RandRange(-4000.0f, 4000.0f);
	float z = FMath::RandRange(3000.0f, 10000.0f);

	FVector location(
		playerLocation.X + x, 
		playerLocation.Y + y, 
		playerLocation.Z + z
	);

	FRotator rotation;
	spawnedActor->SetActorLocationAndRotation(
		location, 
		rotation, 
		false, 
		0, 
		ETeleportType::ResetPhysics
	);

	// We don't want to keep the actor around indefinitely since they may pile up unused.
	// Optionally, we could maintain a queue and cull the oldest actors.
	spawnedActor->StartDestroyTimer(100.0);

	return spawnedActor;
}

ASpawnableSkeletalMeshActor* ASpawningActor::SpawnSkeletalMeshActor()
{
	FVector playerLocation = LookupPlayerLocation();

	// Teleport actor randomly
	// TODO(bt): Put this in front of the camera, or on the (camera -> player) arc!
	float x = FMath::RandRange(-4000.0f, 4000.0f);
	float y = FMath::RandRange(-4000.0f, 4000.0f);
	float z = FMath::RandRange(3000.0f, 10000.0f);

	FVector location(
		playerLocation.X + x, 
		playerLocation.Y + y, 
		playerLocation.Z + z
	);

	return SpawnSkeletalMeshActorAtLocation(location);
}

ASpawnableSkeletalMeshActor* ASpawningActor::SpawnSkeletalMeshActorAtLocation(FVector location)
{
	FActorSpawnParameters actorSpawnParams;
	actorSpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

	FRotator rotation;

	ASpawnableSkeletalMeshActor* spawnedActor = 
		(ASpawnableSkeletalMeshActor*)GetWorld()->SpawnActor(
			ASpawnableSkeletalMeshActor::StaticClass(),
			&location,
			&rotation,
			actorSpawnParams
		);

	// We don't want to keep the actor around indefinitely since they may pile up unused.
	// Optionally, we could maintain a queue and cull the oldest actors.
	spawnedActor->StartDestroyTimer(100.0);

	return spawnedActor;
}

ASoundSpawnActor* ASpawningActor::SpawnSoundActor()
{
	FVector playerLocation = LookupPlayerLocation();

	// Teleport actor randomly
	// TODO(bt): Put this in front of the camera, or on the (camera -> player) arc!
	float x = FMath::RandRange(-1000.0f, 1000.0f);
	float y = FMath::RandRange(-1000.0f, 1000.0f);
	float z = FMath::RandRange(50.0f, 50.0f);

	FVector location(
		playerLocation.X + x, 
		playerLocation.Y + y, 
		playerLocation.Z + z
	);

	return SpawnSoundActorAtLocation(location);
}

ASoundSpawnActor* ASpawningActor::SpawnSoundActorAtLocation(FVector location)
{
	FActorSpawnParameters actorSpawnParams;
	FRotator rotation;

	ASoundSpawnActor* actor = 
		(ASoundSpawnActor*)GetWorld()->SpawnActor(
			ASoundSpawnActor::StaticClass(),
			&location,
			&rotation,
			actorSpawnParams
		);

	return actor;
}

FVector ASpawningActor::LookupPlayerLocation()
{
	return GetWorld()->GetFirstPlayerController()->GetPawn()->GetActorLocation();
}

