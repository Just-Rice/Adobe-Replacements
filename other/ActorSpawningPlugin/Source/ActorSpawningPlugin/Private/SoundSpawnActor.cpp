#include "SoundSpawnActor.h"

#include "Components/AudioComponent.h"
#include "Components/BillboardComponent.h"
#include "Kismet/GameplayStatics.h"
#include "UObject/Object.h"

DEFINE_LOG_CATEGORY(SoundSpawnActorLog);

ASoundSpawnActor::ASoundSpawnActor(const FObjectInitializer& ObjectInitializer) :
	Super(ObjectInitializer),
	BillboardComponent(nullptr),
	HasSoundStarted(false),
	DoDestroyAfterSoundPlays(false)
{
	UE_LOG(SoundSpawnActorLog, Log, TEXT("CTOR()"));

    BillboardComponent = CreateDefaultSubobject<UBillboardComponent>(TEXT("Billboard"));
    BillboardComponent->SetupAttachment(RootComponent);

	AudioComponent = CreateDefaultSubobject<UAudioComponent>(TEXT("Audio"));
    AudioComponent->SetupAttachment(BillboardComponent);

	// We use tick to kill the actor when the sound is done playing
    PrimaryActorTick.bCanEverTick = true;
}

ASoundSpawnActor::~ASoundSpawnActor()
{
}

void ASoundSpawnActor::BeginPlay()
{
	Super::BeginPlay();
}

void ASoundSpawnActor::BeginDestroy()
{
	Super::BeginDestroy();
}

void ASoundSpawnActor::SetSound(USoundBase* sound)
{
	AudioComponent->SetSound(sound);
}

void ASoundSpawnActor::SetSprite(UTexture2D* sprite)
{
	BillboardComponent->SetSprite(sprite);
	BillboardComponent->RegisterComponent();
	BillboardComponent->bHiddenInGame = false; // NB: This defaults to true! :(
}

void ASoundSpawnActor::PlaySoundWithSprite(USoundBase* sound, UTexture2D* sprite, bool DestroyAfterPlay)
{
	SetSprite(sprite);
	SetSound(sound);

	AudioComponent->Play(0.0);

	HasSoundStarted = true;
	DoDestroyAfterSoundPlays = DestroyAfterPlay;
}

void ASoundSpawnActor::PlaySoundImmediate(USoundBase* sound)
{
	UE_LOG(SoundSpawnActorLog, Log, TEXT("playing sound..."));

	UGameplayStatics::PlaySound2D(
		this, // world context
		sound,
		1.0, // VolumeMultiplier
		1.0, // PitchMultiplier 
		0.0 // StartTime
	);
}

void ASoundSpawnActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (HasSoundStarted 
		&& DoDestroyAfterSoundPlays 
		&& !AudioComponent->IsPlaying()) {
		Destroy();
	}
}

