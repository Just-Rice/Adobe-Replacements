#include "LocationLibrary.h"
#include "Kismet/GameplayStatics.h"

FVector ULocationLibrary::GetPlayerAbsoluteLocation(UObject* worldContext)
{
	APawn* playerPawn = GetPlayerPawn(worldContext);
	return playerPawn->GetActorLocation();
}

FVector ULocationLibrary::GetPlayerAbsoluteLocationWithOffset(UObject* worldContext, 
	float x, float y, float z)
{
	FVector playerLocation = GetPlayerAbsoluteLocation(worldContext);
	FVector newLocation(
		playerLocation.X + x, 
		playerLocation.Y + y, 
		playerLocation.Z + z
	);
	return newLocation;
}

FVector ULocationLibrary::GetPlayerAbsoluteLocationWithRandomOffset(UObject* worldContext,
	float minX, float maxX, float minY, float maxY, float minZ, float maxZ)
{
	float x = FMath::RandRange(minX, maxX);
	float y = FMath::RandRange(minY, maxY);
	float z = FMath::RandRange(minZ, maxZ);

	return GetPlayerAbsoluteLocationWithOffset(worldContext, x, y, z);
}


FVector ULocationLibrary::GetPlayerForwardLocationWithOffset(UObject* worldContext,
	float forwardOffset,
	float xAbsoluteOffset, float yAbsoluteOffset, float zAbsoluteOffset)
{
	APawn* playerPawn = GetPlayerPawn(worldContext);
	FVector playerLocation = playerPawn->GetActorLocation();
	FVector playerForwardVector = playerPawn->GetActorForwardVector();

	FVector transformedLocation(
		playerLocation.X + xAbsoluteOffset, 
		playerLocation.Y + yAbsoluteOffset, 
		playerLocation.Z + zAbsoluteOffset
	);

	FVector scaledForwardVector = playerForwardVector * forwardOffset;

	FVector finalLocation = transformedLocation + scaledForwardVector;

	return finalLocation;
}

FVector ULocationLibrary::GetPlayerForwardLocationWithRandomOffset(UObject* worldContext,
	float minForwardOffset, float maxForwardOffset,
	float minAbsoluteXOffset, float maxAbsoluteXOffset,
	float minAbsoluteYOffset, float maxAbsoluteYOffset,
	float minAbsoluteZOffset, float maxAbsoluteZOffset)
{

	float forward = FMath::RandRange(minForwardOffset, maxForwardOffset);
	float x = FMath::RandRange(minAbsoluteXOffset, maxAbsoluteXOffset);
	float y = FMath::RandRange(minAbsoluteYOffset, maxAbsoluteYOffset);
	float z = FMath::RandRange(minAbsoluteZOffset, maxAbsoluteZOffset);

	return GetPlayerForwardLocationWithOffset(worldContext, forward, x, y, z);
}


APawn* ULocationLibrary::GetPlayerPawn(UObject* worldContext)
{
	APawn* playerPawn = UGameplayStatics::GetPlayerPawn(worldContext, 0);
	return playerPawn;
}
