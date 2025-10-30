"use server"

import { db } from "~/server/db";
import { auth } from "~/lib/auth";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { inngest } from "~/inngest/client";

export async function queueSong() {
    // Call the server action to generate a song
    const session = await auth.api.getSession({
        headers: await headers(),
    });
    if (!session) {
        redirect("/auth/sign-in");
    }

    const song = await db.song.create({
        data: {
            userId: session.user.id,
            title: "AI Generated Song",
            fullDescribedSong: "A calming piano melody with soft strings in the background",
        },
    });
    await inngest.send({
        name: "generate-song-event",
        data: {
            songId: song.id,
            userId: session.user.id,
        },
    });
}