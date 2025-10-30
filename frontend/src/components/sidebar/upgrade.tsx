"use client";

import { authClient } from "~/lib/auth-client";
import { Button } from "../ui/button";

export default function Upgrade() {
  const upgrade = async () => {
    await authClient.checkout({
      products: [
        "db9bb706-bd30-498a-9a56-03c5e38f19cf",
        "ab5e1d31-2955-4317-885c-bdc0b5596838",
        "89a4e433-26b7-4bd1-b9c0-f7281fcc0399"
      ],
    });
  };
  return (
    <Button
      variant="outline"
      size="sm"
      className="ml-2 cursor-pointer text-orange-400"
      onClick={upgrade}
    >
      Upgrade
    </Button>
  );
}
