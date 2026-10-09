#pragma once

// Unreal Core
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Engine/EngineTypes.h"

// EventCenterPlugin
#include "SoundSpawnActor.generated.h"

DECLARE_LOG_CATEGORY_EXTERN(SoundSpawnActorLog, Log, All);

UCLASS()
class ACTORSPAWNINGPLUGIN_API ASoundSpawnActor : public AActor
{
	GENERATED_BODY()

public:

	ASoundSpawnActor(const FObjectInitializer& ObjectInitializer);
	~ASoundSpawnActor();

	/** Access to the audio component */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SoundSpawnActor")
	class UAudioComponent* AudioComponent;

	/** Access to the billboard component */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "SoundSpawnActor")
	class UBillboardComponent* BillboardComponent;

	/** Set sound to the audio component. */
	UFUNCTION(BlueprintCallable, Category = "SoundSpawnActor")
	void SetSound(USoundBase* sound);

	/** Set a sprite. */
	UFUNCTION(BlueprintCallable, Category = "SoundSpawnActor")
	void SetSprite(UTexture2D* sprite);

	/** Play sound with the sprite, then destroy. */
	UFUNCTION(BlueprintCallable, Category = "SoundSpawnActor")
	void PlaySoundWithSprite(USoundBase* sound, UTexture2D* sprite, bool DestroyAfterPlay);

	/** Play a sound. Does not set to the component. */
	UFUNCTION(BlueprintCallable, Category = "SoundSpawnActor")
	void PlaySoundImmediate(USoundBase* sound);

	// Called every frame
	// Used to invoke destroy.
	virtual void Tick(float DeltaTime) override;

protected:
	// Whether sound has started playing at any point.
	bool HasSoundStarted;

	// Whether to destroy after sound finishes playing the first time.
	bool DoDestroyAfterSoundPlays;

	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	virtual void BeginDestroy() override;
};

