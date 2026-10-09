#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Engine/EngineTypes.h"

// EventCenterPlugin
#include "SpawnableActor.h"
#include "SpawnableSkeletalMeshActor.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(SpawnableSkeletalMeshActorLog, Log, All);

// NB(bt): "Blueprintable" seems to remove the ability to add BP event nodes.
UCLASS()
class ACTORSPAWNINGPLUGIN_API ASpawnableSkeletalMeshActor : public ASpawnableActor
{
	GENERATED_BODY()

public:

	ASpawnableSkeletalMeshActor();
	~ASpawnableSkeletalMeshActor();

	/** Access to the skeletal mesh component */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SpawnableSkeletalMeshActor", meta = (AllowPrivateAccess = "true"))
	class USkeletalMeshComponent* BaseMeshComponent;

	/** Set the skeletal mesh. */
	UFUNCTION(BlueprintCallable, Category = "SpawnableSkeletalMeshActor")
	void SetSpawnableSkeletalMesh(USkeletalMesh* skeletalMesh);

private:
	/** Mesh to use */
	//UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SpawnableActor", meta = (AllowPrivateAccess = "true"))
	class USkeletalMesh* BaseMesh;
};

